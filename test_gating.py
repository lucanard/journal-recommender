"""Checks for the server-side gating of the paid endpoints.

Run with:  python test_gating.py

These cover the decisions that used to live in the browser — who is signed in,
how many results come back, whether a credit is spent — plus the Stripe redirect
URLs, which previously sent every paying customer to a 404.

/recommend is not exercised end to end: the vector store holds Gemini embeddings
and a real run needs an API key. The engine is stubbed instead and we assert on
what the endpoint asks it for, which is exactly the part a client used to
control. No network calls, no credentials, no Firebase project required.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PUBLIC_BASE_URL", "https://pubfit.ai")
sys.argv = ["app.py"]
sys.path.insert(0, HERE)
os.chdir(HERE)

FAILURES = []


def check(label, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + label + ("" if condition else f"   -> {detail}"))
    if not condition:
        FAILURES.append(label)


class StubEngine:
    """Stands in for the recommendation engine and records what it was asked."""
    llm = object()

    def __init__(self):
        self.calls = []

    def recommend(self, abstract, constraints, num_results):
        self.calls.append({"num_results": num_results, "constraints": constraints})
        return {
            "recommendations": [], "analysis_summary": "stub", "timing": {},
            "candidates_searched": 0, "candidates_after_filter": 0,
        }


ABSTRACT = (
    "We study deep learning models for the automated detection of diabetic "
    "retinopathy in retinal fundus images using convolutional networks."
)

PREMIUM_REQUEST = {
    "abstract": ABSTRACT, "num_results": 10,
    "indexing_required": ["Scopus"], "oa_preference": "Open Access Only",
    "apc_free_only": True, "max_apc": 500, "min_impact_factor": 5.0,
    "target_impact": "Q1 (High)",
}


def test_anonymous_is_refused(app_module, client):
    """With enforcement on, the paid endpoints need a verified token."""
    app_module.AUTH_ENFORCED = True
    for path, payload in [
        ("/recommend", {"abstract": ABSTRACT}),
        ("/tailor", {"abstract": ABSTRACT, "journal_id": 32}),
        ("/report", {"results": {}, "abstract": "x"}),
        ("/create-checkout-session", {"pack_id": "starter", "uid": "x"}),
    ]:
        r = client.post(path, json=payload)
        check(f"anonymous: {path} refused", r.status_code == 401, r.status_code)

    r = client.post("/recommend", headers={"Authorization": "Bearer forged.token"},
                    json={"abstract": ABSTRACT})
    check("anonymous: forged token refused", r.status_code == 401, r.status_code)

    for path in ["/health", "/stats", "/disciplines"]:
        check(f"public: {path} still open", client.get(path).status_code == 200)


def test_free_tier_is_capped(app_module, client):
    """A caller with no credits gets the free tier however the request is framed."""
    app_module.AUTH_ENFORCED = False
    app_module._verify_bearer_token = lambda rq: None
    engine = StubEngine()
    app_module.engine = engine

    r = client.post("/recommend", json=PREMIUM_REQUEST)
    check("free tier: accepted", r.status_code == 200, r.text[:200])
    call = engine.calls[-1]
    constraints = call["constraints"]
    check("free tier: 10 results asked, 3 delivered", call["num_results"] == 3, call["num_results"])
    check("free tier: indexing filter stripped", constraints.indexing_required == [])
    check("free tier: OA filter stripped", constraints.oa_preference == "Any")
    check("free tier: APC-free filter stripped", constraints.apc_free_only is False)
    check("free tier: max APC stripped", constraints.max_apc is None)
    check("free tier: min impact factor stripped", constraints.min_impact_factor is None)
    check("free tier: target impact stripped", constraints.target_impact is None)
    check("free tier: response says so", r.json().get("tier") == "free")


def test_premium_tier_is_honoured(app_module, client):
    """A caller holding a credit keeps every filter and the full result count."""
    app_module._verify_bearer_token = lambda rq: "uid-with-credits"
    app_module._spend_search_credit = lambda uid: True
    app_module._read_credits = lambda uid: 4
    engine = StubEngine()
    app_module.engine = engine

    r = client.post("/recommend", headers={"Authorization": "Bearer valid"}, json=PREMIUM_REQUEST)
    check("premium: accepted", r.status_code == 200, r.text[:200])
    call = engine.calls[-1]
    constraints = call["constraints"]
    check("premium: all 10 results delivered", call["num_results"] == 10, call["num_results"])
    check("premium: indexing filter preserved", constraints.indexing_required == ["Scopus"])
    check("premium: OA filter preserved", constraints.oa_preference == "Open Access Only")
    check("premium: APC-free preserved", constraints.apc_free_only is True)
    check("premium: max APC preserved", constraints.max_apc == 500)
    check("premium: min impact factor preserved", constraints.min_impact_factor == 5.0)
    check("premium: target impact preserved", constraints.target_impact == "Q1 (High)")
    body = r.json()
    check("premium: response says so", body.get("tier") == "premium")
    check("premium: remaining balance reported", body.get("credits_remaining") == 4)


def test_failed_search_refunds_the_credit(app_module, client):
    """The credit is taken before the search runs, so a crash has to give it back."""
    refunded = []
    app_module._refund_search_credit = lambda uid: refunded.append(uid)

    class ExplodingEngine(StubEngine):
        def recommend(self, *args, **kwargs):
            raise RuntimeError("engine exploded")

    app_module.engine = ExplodingEngine()
    r = client.post("/recommend", headers={"Authorization": "Bearer valid"},
                    json={"abstract": ABSTRACT})
    check("refund: failed search returns 500", r.status_code == 500, r.status_code)
    check("refund: credit given back", refunded == ["uid-with-credits"], refunded)


def test_credit_arithmetic(app_module):
    """Expiry, legacy accounts and malformed data all resolve to a safe balance."""
    future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    cases = [
        ("live credits count", {"credits": 7, "credits_expire_at": future}, 7),
        ("expired credits are worth nothing", {"credits": 7, "credits_expire_at": past}, 0),
        ("a missing expiry counts as expired", {"credits": 7}, 0),
        ("legacy premium account honoured", {"plan": "premium"}, 60),
        ("legacy free account gets nothing", {"plan": "free"}, 0),
        ("negative balance floored at zero", {"credits": -5, "credits_expire_at": future}, 0),
        ("unparseable expiry counts as expired", {"credits": 3, "credits_expire_at": "nope"}, 0),
    ]
    for label, data, expected in cases:
        actual = app_module._effective_credits(data)
        check(f"balance: {label}", actual == expected, f"got {actual}, wanted {expected}")


def test_rate_limit(app_module, client):
    """The per-minute budget closes the door on a script hammering the endpoint."""
    app_module.RATE_LIMIT_PER_MINUTE = 5
    app_module._rate_hits.clear()
    app_module.engine = StubEngine()
    codes = [client.post("/recommend", headers={"Authorization": "Bearer valid"},
                         json={"abstract": ABSTRACT}).status_code for _ in range(8)]
    check("rate limit: first calls allowed", codes[0] == 200, codes)
    check("rate limit: 429 once the budget is spent", 429 in codes, codes)
    app_module.RATE_LIMIT_PER_MINUTE = 0
    app_module._rate_hits.clear()


def test_checkout(app_module, client):
    """Checkout trusts the token, and sends the customer back to a page that exists."""
    import stripe
    captured = {}

    class FakeSession:
        id = "cs_test_stub"
        url = "https://checkout.stripe.com/stub"

    def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeSession()

    app_module.STRIPE_AVAILABLE = True
    os.environ["STRIPE_SECRET_KEY"] = "sk_test_stub"
    original = stripe.checkout.Session.create
    stripe.checkout.Session.create = staticmethod(fake_create)
    try:
        r = client.post("/create-checkout-session", headers={"Authorization": "Bearer valid"},
                        json={"pack_id": "starter", "uid": "SOMEONE-ELSES-UID", "email": "a@b.c"})
        check("checkout: session created", r.status_code == 200, r.text[:200])
        check("checkout: body uid ignored in favour of the token",
              captured.get("client_reference_id") == "uid-with-credits",
              captured.get("client_reference_id"))
        check("checkout: metadata uid comes from the token too",
              (captured.get("metadata") or {}).get("uid") == "uid-with-credits",
              captured.get("metadata"))
        success = captured.get("success_url", "")
        check("checkout: success URL puts the query before the hash",
              success.startswith("https://pubfit.ai/?purchase=success") and success.endswith("#/dashboard"),
              success)
        check("checkout: cancel URL puts the query before the hash",
              captured.get("cancel_url") == "https://pubfit.ai/?purchase=cancelled#/pricing",
              captured.get("cancel_url"))
    finally:
        stripe.checkout.Session.create = original


def main():
    try:
        import app as app_module
        from fastapi.testclient import TestClient
    except ImportError as e:
        print(f"Cannot run: {e}\nInstall the deps first: pip install -r requirements-deploy.txt httpx")
        return 2

    client = TestClient(app_module.app)
    app_module.RATE_LIMIT_PER_MINUTE = 0  # off unless a test turns it on

    test_anonymous_is_refused(app_module, client)
    test_free_tier_is_capped(app_module, client)
    test_premium_tier_is_honoured(app_module, client)
    test_failed_search_refunds_the_credit(app_module, client)
    test_credit_arithmetic(app_module)
    test_rate_limit(app_module, client)
    test_checkout(app_module, client)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
