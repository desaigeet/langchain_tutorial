from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str = "Sunny"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(ge=0, lt=10, default=0.0, description="Decimal Value representing cgpa of student, must be between 0 and 10")

newStudent = {
    "name": "Geet Desai",
    "age": "28",
    "email": "geet123@gmail.com",
    "cgpa" : 9.6
}

student = Student(**newStudent)

print(student)
