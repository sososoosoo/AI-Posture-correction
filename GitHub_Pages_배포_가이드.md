# 🚀 GitHub Pages 배포 가이드

## 📋 준비물
- GitHub 계정
- `index.html` 파일 (이미 준비됨)
- 배포할 실행파일들 (optional)

---

## 1️⃣ GitHub 리포지토리 생성

### 1. GitHub에서 새 리포지토리 만들기
```
1. GitHub.com 로그인
2. 우측 상단 "+" 버튼 클릭
3. "New repository" 선택
4. Repository name: "AI-posture" (또는 원하는 이름)
5. Public 선택 (중요!)
6. "Create repository" 클릭
```

---

## 2️⃣ 로컬에서 Git 설정

### 현재 폴더에서 Git 초기화
```bash
cd C:\AI_solution\camera
git init
git add Posture_correction.html
git commit -m "Add AI posture analysis website"
```

### GitHub 리포지토리와 연결
```bash
git remote add origin https://github.com/사용자명/AI-posture.git
git branch -M main
git push -u origin main
```

---

## 3️⃣ GitHub Pages 활성화

### GitHub 웹사이트에서 설정
```
1. 생성한 리포지토리로 이동
2. "Settings" 탭 클릭
3. 왼쪽 사이드바에서 "Pages" 클릭
4. Source: "Deploy from a branch" 선택
5. Branch: "main" 선택
6. Folder: "/ (root)" 선택
7. "Save" 클릭
```

### 🎉 완료!
- 5-10분 후 웹사이트 접속 가능
- 주소: `https://사용자명.github.io/AI-posture/`

---

## 4️⃣ 실행파일 배포 (Release 방식)

### 실행파일을 GitHub Release로 업로드

```
1. 리포지토리에서 "Releases" 클릭
2. "Create a new release" 클릭
3. Tag version: "v1.0.0" 입력
4. Release title: "AI 자세분석 v1.0.0" 입력
5. 설명 작성:
```

```markdown
## 🚀 AI 자세분석 프로그램 v1.0.0

### 주요 기능
- 실시간 자세 분석
- AI 기반 자세 교정 피드백
- 엑셀 리포트 저장

### 다운로드
- Windows 10/11 지원
- 4GB RAM 이상 권장

### 설치 방법
1. AI_자세분석.exe 다운로드
2. 관리자 권한으로 실행
3. 설치 완료 후 프로그램 실행
```

```
6. "Attach binaries" 영역에 .exe 파일 드래그 앤 드롭
7. "Publish release" 클릭
```

---

## 5️⃣ 웹사이트에서 다운로드 링크 수정

### index.html에서 링크 수정 필요
현재 코드에서 이 부분들을 수정:

```javascript
// 수정 전
window.open('https://github.com/yourname/AI-posture/releases/latest', '_blank');

// 수정 후 (실제 사용자명으로 변경)
window.open('https://github.com/실제사용자명/AI-posture/releases/latest', '_blank');
```

```html
<!-- 수정 전 -->
<a href="https://github.com/yourname/AI-posture/releases" className="release-link">

<!-- 수정 후 -->
<a href="https://github.com/실제사용자명/AI-posture/releases" className="release-link">
```

### 수정 후 업데이트
```bash
git add Posture_correction.html
git commit -m "Update download links"
git push origin main
```

---

## 6️⃣ 폴더 구조 (최종)

```
AI-posture/
├── index.html                 # 메인 웹사이트
├── README.md                  # 프로젝트 설명
├── .gitignore                 # Git 제외 파일들
└── releases/                  # (optional) 로컬 백업용
    ├── AI_자세분석_v1.0.exe
    └── AI_자세분석_mobile.apk
```

---

## 7️⃣ 추가 설정 (선택사항)

### 커스텀 도메인 설정
```
1. 도메인 구입 후 DNS 설정
2. GitHub Pages Settings에서 Custom domain 입력
3. CNAME 파일 생성 (자동 생성됨)
```

### SEO 최적화
```html
<!-- index.html에 이미 포함됨 -->
<meta name="description" content="...">
<meta property="og:title" content="...">
<meta property="og:description" content="...">
```

---

## 🔧 문제 해결

### 사이트가 안 뜰 때
1. GitHub Pages 활성화 확인
2. main 브랜치에 index.html 있는지 확인
3. 5-10분 대기 (배포 시간)

### 파일이 업데이트 안 될 때
```bash
# 브라우저 캐시 삭제 또는
# Ctrl + F5 (강제 새로고침)
```

### 다운로드 링크가 안 될 때
- GitHub Release가 Public인지 확인
- 파일이 올바르게 업로드되었는지 확인

---

## 📱 결과 확인

### 최종 확인 사항
- ✅ 웹사이트: `https://사용자명.github.io/AI-posture/`
- ✅ 다운로드: `https://github.com/사용자명/AI-posture/releases`
- ✅ 모바일 반응형 동작
- ✅ 모든 탭 정상 작동
- ✅ 애니메이션 효과 정상

---

## 🎯 다음 단계

1. **도메인 연결** (선택): 커스텀 도메인 구입 및 연결
2. **Analytics 추가**: Google Analytics 연동
3. **업데이트**: 새 버전 Release로 지속 업데이트
4. **SEO 개선**: sitemap.xml, robots.txt 추가

---

💡 **팁**: GitHub Actions를 사용하면 자동 배포도 설정할 수 있어요!