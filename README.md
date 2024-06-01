# DS_termproject

### 1) misisng data 확인
- bmi/ smole status 에서 발견

### 2) missing data / outiler 수정
1. bmi
- Random forest 사용하여 값 채우기

 파일 :
```
bmi_forest.py
```
-> 결과 : filled_bmi 파일  
***

2. smoke_status
- no info 라는 카테고리 생성
파일 :
```
final_data.py
```
-> 결과 : final_data 파일
***

3. outiler
- age와 bmi에 outlier 존재
  -> IQR사용해 Q1보다 작음 Q1로 값 변환, Q3보다 크면 Q3으로 값 변환





