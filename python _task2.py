d1={'Rahul':100,'mohan':78,'Shubham':45,'Nikhil':34,'Niyati':65}
def above50(d1):
    d2={}
    for i in d1:
        if d1[i]>50:
            d2.update({i:d1[i]})
    print(d2)
above50(d1) 