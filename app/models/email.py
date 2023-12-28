from pydantic import BaseModel, EmailStr
from typing import List
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import smtplib
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv(os.path.join(os.path.dirname(__file__), "email.env"))

class EmailSchema(BaseModel):
    to: str
    subject: str
    body: str


def send_email(to_addresses, subject, body):
    from_email = os.getenv('FROM_EMAIL')
    email_password = os.getenv('EMAIL_PASSWORD')

    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = ",".join(to_addresses)
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(from_email, email_password)
        server.sendmail(from_email, to_addresses, message.as_string())

