# Vendored front-end assets

Third-party scripts and fonts used by `index.html`, served from the app's own
origin by `app.py` (`GET /vendor/{name}`) so that loading the page sends the
visitor's IP address to no CDN or font host (GDPR; LG München I, 20.01.2022,
3 O 17493/20, on Google Fonts). Files are byte-identical to upstream and keep
their licence headers. Added 2026-10-19.

Served: the `.js` and `.woff2` files directly in this folder at
`/vendor/<name>`, and the `.txt` licence texts in `licenses/` at
`/vendor/licenses/<name>`. Nothing else (this file included) is reachable.

## Files

| File | Package and version | Source | SHA-384 (base64) | Licence |
|---|---|---|---|---|
| `tailwindcss-play-cdn-3.4.17.js` | Tailwind CSS Play CDN 3.4.17 | https://cdn.tailwindcss.com/3.4.17 (what `https://cdn.tailwindcss.com` redirected to on 2026-10-19) | `igm5BeiBt36UU4gqwWS7imYmelpTsZlQ45FZf+XBn9MuJbn4nQr7yx1yFydocC/K` | MIT, `licenses/tailwindcss-MIT.txt` |
| `react-18.3.1.production.min.js` | react 18.3.1, `umd/react.production.min.js` | npm tarball `react-18.3.1.tgz` | `DGyLxAyjq0f9SPpVevD6IgztCFlnMF6oW/XQGmfe+IsZ8TqEiDrcHkMLKI6fiB/Z` | MIT, `licenses/react-MIT.txt` |
| `react-dom-18.3.1.production.min.js` | react-dom 18.3.1, `umd/react-dom.production.min.js` | npm tarball `react-dom-18.3.1.tgz` | `gTGxhz21lVGYNMcdJOyq01Edg0jhn/c22nsx0kyqP0TxaV5WVdsSH1fSDUf5YJj1` | MIT, `licenses/react-MIT.txt` |
| `babel-standalone-7.23.7.min.js` | @babel/standalone 7.23.7, `babel.min.js` | npm tarball `babel-standalone-7.23.7.tgz` | `qR8cCjBLKaeE7HdZ1ixWruJRkc5giIbp6Mmva6jZW0oNATuhO7uTDR9Tm/aFO6WN` | MIT, `licenses/babel-standalone-MIT.txt` |
| `firebase-app-compat-10.14.1.js` | firebase 10.14.1, `firebase-app-compat.js` | npm tarball `firebase-10.14.1.tgz`; identical to https://www.gstatic.com/firebasejs/10.14.1/firebase-app-compat.js | `ZaR6mWzmJtrRibZ1Vm7SoHFr8OXjyAuGAXalGDKqbxFT18oi/z+oZLIRFkpeNor1` | Apache-2.0, `licenses/firebase-Apache-2.0.txt` |
| `firebase-auth-compat-10.14.1.js` | firebase 10.14.1, `firebase-auth-compat.js` | as above; identical to the gstatic copy | `I1LYojsZ5RM1cOda44Z2h42Qa6YfsQ1XkXxREnhp4ueYBR/4d1pG1K+NZM537Vsj` | Apache-2.0 |
| `firebase-firestore-compat-10.14.1.js` | firebase 10.14.1, `firebase-firestore-compat.js` | as above; identical to the gstatic copy | `Ke0FJhH7LyRqDxZ0wt+/OXV38yfQVu7g9VPEEGjYmB4RVOY/ta04uecRhsMwT7V3` | Apache-2.0 |
| `dm-sans-4.004-normal-latin.woff2` | DM Sans 4.004 variable (opsz 9–40, wght 100–1000), normal, latin subset | copied from the pubfit-website repo, `public/assets/fonts/dm-sans-normal-latin.woff2` (Google Fonts subset) | `H+4W4FHMKbn0A/IxuKmnoB5/G0F1At8yw1MDI3quP1b3xQfJTq6ltYKW/YnMoUj9` | SIL OFL 1.1, `licenses/dm-sans-OFL-1.1.txt` |
| `dm-sans-4.004-normal-latin-ext.woff2` | as above, latin-ext subset | `public/assets/fonts/dm-sans-normal-latin-ext.woff2` | `NC/aMzKWL//tfAkFPOXRx6eU+hDKI//kgtmxcbAqeRZorumogf38Eq5POLM3ar3R` | SIL OFL 1.1 |
| `dm-sans-4.004-italic-latin.woff2` | as above, italic, latin subset | `public/assets/fonts/dm-sans-italic-latin.woff2` | `tq94fKqeR5avSCRSpIPyXVUa9xhQN8sCxZFQl8aYHKlyflzvxp/U8NvHm/hS101k` | SIL OFL 1.1 |
| `dm-sans-4.004-italic-latin-ext.woff2` | as above, italic, latin-ext subset | `public/assets/fonts/dm-sans-italic-latin-ext.woff2` | `ITsU8aLqEhxG2lClFqnDEyeFuR47IJL73sob6A/FK13pQK+5GtCwns4FjyDdoJJR` | SIL OFL 1.1 |

npm tarballs were fetched with `npm pack <pkg>@<version> --ignore-scripts`,
which checks them against the registry's integrity value:

| Tarball | npm `dist.integrity` |
|---|---|
| react-18.3.1.tgz | `sha512-wS+hAgJShR0KhEvPJArfuPVN1+Hz1t0Y6n5jLrGQbkb4urgPE/0Rve+1kMB1v/oWgHgm4WIcV+i7F2pTVj+2iQ==` |
| react-dom-18.3.1.tgz | `sha512-5m4nQKp+rZRb09LNH59GM4BxTh9251/ylbKIbpe7TpGxfJ+9kv6BLkLBXIjjspbgbnIBNqlI23tRnTWT0snUIw==` |
| babel-standalone-7.23.7.tgz | `sha512-AsO3aIh9I4qIqK61d6nPxPAdrSuWF4FmOLej3xNIkBIZj+8XJGArQQJw6DnuUkkqbsLp1fARkXOdKiuqWgac0Q==` |
| firebase-10.14.1.tgz | `sha512-0KZxU+Ela9rUCULqFsUUOYYkjh7OM1EWdIfG6///MtXd0t2/uUIf0iNV5i0KariMhRQ5jve/OY985nrAXFaZeQ==` |

The Tailwind Play CDN build is not published on npm, so it has no registry
hash; the SHA-384 above is of the file as served on 2026-10-19.

The firebase npm package ships no LICENSE file (its `package.json` declares
Apache-2.0); the licence text here is the standard Apache License 2.0. The
pubfit-website repo has no OFL file for DM Sans; the OFL text here is the
standard SIL OFL 1.1 with the copyright line read from the fonts' name table.

Advisory check on 2026-10-19 (OSV, covering GitHub Advisories and NVD aliases):
no known vulnerabilities for react 18.3.1, react-dom 18.3.1, @babel/standalone
7.23.7, firebase 10.14.1 (@firebase/app 0.10.13, app-compat 0.2.43, auth 1.7.9,
auth-compat 0.5.14, firestore 4.7.3, firestore-compat 0.3.38) or tailwindcss
3.4.17.

## Upgrading a file

1. Put the new version in this folder under a new versioned name (never reuse
   a name: browsers cache each file for a year as immutable).
2. Update the `src` and the `integrity` attribute in `index.html`:
   `openssl dgst -sha384 -binary <file> | openssl base64 -A`.
3. Update the table above, then delete the old file.

## Not vendored, still contacted at runtime

- Tally (`https://tally.so/widgets/embed.js` and the form iframe): loaded only
  after the visitor clicks "Open the enquiry form" on the pricing page.
- Firebase Auth and Firestore APIs (`*.googleapis.com`), the sign-in helper on
  `pubfit-1a5ad.firebaseapp.com`, and the Google script the Auth SDK can load
  for popup sign-in (`apis.google.com/js/api.js`): runtime APIs used when the
  visitor signs in or uses their account, not static assets. None of them is
  contacted on page load (checked 2026-10-19).
