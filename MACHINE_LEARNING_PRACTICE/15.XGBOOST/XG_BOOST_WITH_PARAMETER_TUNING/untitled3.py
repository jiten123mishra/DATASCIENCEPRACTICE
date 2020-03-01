s="promotion_type,promotion_type_description"
l=s.split(",")
out="where"
for i in range (len(l)):
    out=out+" "+l[i]+"is not null"
print(out)