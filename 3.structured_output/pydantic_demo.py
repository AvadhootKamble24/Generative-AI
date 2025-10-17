from pydantic import BaseModel, EmailStr,Field
from typing import Optional

class Student (BaseModel):
    name: str = 'Avadhoot'
    age: Optional[int]=None
    email: EmailStr
    cgpa: float=Field(gt=0, lt=10,default=5,description='A decimal value representing the cgpa of the student')

new_student={
    'age':22,
    'email': 'kavadhoot1234@gmail.com'
    }

student=Student(**new_student)

# studnet_dict= dict(student)
student_dict=student.model_dump() #can also be used to convert into dictionary

print(student_dict)

student_json=student.model_dump_json()