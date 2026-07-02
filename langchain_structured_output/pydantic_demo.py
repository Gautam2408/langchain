from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'User'      #default value will be User
    age: Optional[int] = None   #age field is optional but it will assign value None when kept optional
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description="A decimal value representing cgpa of student")

new_student = {'name': 'gautam', 'age':32, 'email': '123@newton.co.in', 'cgpa': '9'}

student = Student(**new_student)

print(student)