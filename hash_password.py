from werkzeug.security import generate_password_hash

password = "123"  # 🔁 Replace with the password you want to hash
hashed_pw = generate_password_hash(password)
print("Hashed Password:", hashed_pw)
