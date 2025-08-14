Setup on new pod
1. Copy ssh details into ~/.ssh/config (can do in cursor)
NOTE: make sure to get port > changes on every restart
2. Connect to pod via cursor ssh
3. Setup github creds
  a. ssh-keygen -t ed25519 -C "hirdcasey@gmail.com"
  b. cat ~/.ssh/id_ed25519.pub
  c. copy paste into github ssh keys
4. Setup git config (not needed until committing)
  a. git config --global user.email "hirdcasey@gmail.com"
  b. git config --global user.name "Casey Hird"
5. git clone git@github.com:caseyhird/triton_practice.git
6. cd triton_practice
7. setup uv
  a. pip install uv
  b. uv sync

