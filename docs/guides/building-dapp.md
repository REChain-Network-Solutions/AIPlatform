# Building a DApp on AIPlatform

This guide provides a concise overview of building a decentralized application (DApp) that integrates with AIPlatform's Web3/4/5 stack.

## Project Structure

```
dapp-ai-classifier/
├── contracts/           # Smart contracts
├── backend/            # FastAPI service
└── frontend/          # React frontend
```

## 1. Smart Contracts

### AIToken.sol
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract AIToken is ERC20, Ownable {
    constructor() ERC20("AI Token", "AIT") Ownable(msg.sender) {
        _mint(msg.sender, 1000000 * 10 ** decimals());
    }
    
    function mint(address to, uint256 amount) public onlyOwner {
        _mint(to, amount);
    }
}
```

## 2. Backend (FastAPI)

### main.py
```python
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    return {"class": "cat", "confidence": 0.95}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 3. Frontend (React)

### App.jsx
```jsx
import React, { useState } from 'react';
import { Web5 } from '@web5/api';

function App() {
  const [web5, setWeb5] = useState(null);
  const [did, setDid] = useState('');

  const connectWeb5 = async () => {
    const { web5, did } = await Web5.connect();
    setWeb5(web5);
    setDid(did);
  };

  return (
    <div>
      <button onClick={connectWeb5}>
        {did ? `Connected: ${did.slice(0, 10)}...` : 'Connect Web5'}
      </button>
    </div>
  );
}

export default App;
```

## 4. Web5 Integration

### web5-utils.js
```javascript
export const saveToDWN = async (web5, data) => {
  const { record } = await web5.dwn.records.create({
    data,
    message: {
      dataFormat: 'application/json',
    },
  });
  return record;
};
```

## 5. Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Next Steps

1. Add user authentication
2. Implement token payments
3. Add data contribution features
4. Deploy to decentralized storage
