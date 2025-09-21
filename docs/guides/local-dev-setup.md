# Local Development Setup

This guide will help you set up a local development environment for building on the AIPlatform.

## Prerequisites

- Node.js 18+
- Python 3.8+
- Git
- Docker (optional)
- MetaMask (browser extension)

## 1. Clone the Repository

```bash
git clone https://github.com/REChain-Network-Solutions/AIPlatform.git
cd AIPlatform
```

## 2. Install Dependencies

### Frontend
```bash
cd frontend
npm install
```

### Backend
```bash
cd ../backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Web3 Provider
WEB3_PROVIDER_URL=http://localhost:8545

# Backend
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///./aiplatform.db

# Frontend
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WEB3_NETWORK=localhost
```

## 4. Start Local Blockchain

### Using Hardhat
```bash
npx hardhat node
```

## 5. Deploy Contracts

```bash
npx hardhat run scripts/deploy.js --network localhost
```

## 6. Start Development Servers

### Backend
```bash
cd backend
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm start
```

## 7. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Hardhat Node: http://localhost:8545

## Troubleshooting

### Common Issues

1. **Node.js version mismatch**
   - Use `nvm use` if you have `.nvmrc`
   - Or install the required version: `nvm install 18`

2. **Python virtual environment**
   - Make sure to activate the virtual environment
   - Reinstall requirements if needed: `pip install -r requirements.txt`

3. **Metamask Connection**
   - Make sure you're on the local network (chainId: 31337)
   - Import test accounts from Hardhat

4. **Port conflicts**
   - Check which process is using the port: `lsof -i :3000`
   - Kill the process: `kill -9 <PID>`

## Development Workflow

1. Make changes to the code
2. Run tests: `npm test` (frontend) or `pytest` (backend)
3. Commit changes with semantic messages:
   ```
   feat: add user authentication
   fix: resolve login issue
   docs: update README
   ```
4. Push to your feature branch
5. Create a pull request

## Resources

- [Hardhat Documentation](https://hardhat.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/getting-started.html)
- [Web5 Documentation](https://developer.tbd.website/docs/web5/learn/web5-js-sdk/)
