# mpea.alloys.dev

On the production Debian host, Lighttpd terminates HTTPS and proxies this
hostname to Gunicorn on 127.0.0.1:8173. The app runs as an unprivileged systemd
dynamic user. One synchronous worker fits the host's 1 GiB of RAM. It loads
the model once and caches up to 128 predictions. Restart the service after
changing code or model data.

## Install on the host

Place the repository in /srv/mpea. Install python3-venv and use Python 3.11
or newer. Run these commands as the deployment administrator.

```sh
cd /srv/mpea
python3 -m venv .venv
.venv/bin/pip install --prefer-binary -r requirements.txt
.venv/bin/python -m unittest discover -s tests
install -m 0644 deploy/mpea.service /etc/systemd/system/mpea.service
systemctl daemon-reload
systemctl enable --now mpea
curl --retry 10 --retry-connrefused --retry-delay 2 -fsS http://127.0.0.1:8173/api/AlCo:0.9
```

## Configure HTTPS on first installation

Point the hostname's A record at the server. The existing Lighttpd installation
listens on 80 and 443 with mod_openssl and mod_redirect. Its default HTTP
document root must serve Certbot's challenge files for the new hostname.
Set ACME_WEBROOT to that directory before issuing the certificate.

```sh
certbot certonly --webroot -w "${ACME_WEBROOT:?set the HTTP challenge webroot}" \
    --cert-name mpea.alloys.dev -d mpea.alloys.dev
install -m 0644 deploy/mpea.lighttpd.conf /etc/lighttpd/conf-available/50-mpea.conf
ln -s ../conf-available/50-mpea.conf /etc/lighttpd/conf-enabled/50-mpea.conf
lighttpd -tt -f /etc/lighttpd/lighttpd.conf && systemctl restart lighttpd
install -m 0755 deploy/apply-mpea-certificate /etc/letsencrypt/renewal-hooks/deploy/50-mpea
certbot renew --cert-name mpea.alloys.dev --dry-run
```

The hook restarts Lighttpd after this certificate renews so its master can
read the root-protected keys before dropping privileges.

## Verify and update

```sh
curl -sSI http://mpea.alloys.dev/
curl -fsS https://mpea.alloys.dev/api/AlCo:0.9
curl -fsS https://mpea.alloys.dev/api/Al0.1CoCrFeMnNi:0.9
```

The HTTP request should redirect to HTTPS. AlCo should return HCP, unstable,
and about 0.27835 eV/atom above the hull including intermetallics. The
six-element example should return FCC, stable, and about 1606 K.

For an update, transfer the changed source and assets, install any changed
requirements in the same virtual environment, run the regression test, then
run `systemctl restart mpea` and repeat the public checks. Use
`journalctl -u mpea` for service logs.
