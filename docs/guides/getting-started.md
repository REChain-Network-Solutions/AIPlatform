# Getting Started with AIPlatform Web3/4/5

This guide will help you set up your development environment and start building on the AIPlatform.

## Prerequisites

- Node.js 18+
- npm 9+
- Git
- Python 3.8+ (for AI/ML components)
- Docker (optional, for local development)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/aiplatform.git
   cd aiplatform
   ```

2. **Install dependencies**
   ```bash
   # Install JavaScript dependencies
   npm install
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start development servers**
   ```bash
   # Start local blockchain
   npx hardhat node
   
   # In a new terminal, deploy contracts
   npx hardhat run scripts/deploy.js --network localhost
   
   # Start frontend development server
   npm run dev
   ```

## Project Structure

```
aiplatform/
├── contracts/           # Smart contracts (Solidity)
├── dwn/                 # Decentralized Web Nodes
├── frontend/            # Web interface
├── ai/                  # AI/ML models and training
├── docs/                # Documentation
└── tests/               # Test suites
```

## Core Concepts

### 1. Web5: Identity and Data
- **DID**: Decentralized Identifiers for users and devices
- **DWN**: Personal data storage with access control
- **VCs**: Verifiable Credentials for attestations

### 2. Web4: AI/ML Layer
- Federated learning
- Model marketplaces
- Data processing pipelines

### 3. Web3: Blockchain Layer
- Smart contracts for governance
- Token economics
- Decentralized storage

## Example: Create a Decentralized AI Application

1. **Set up user authentication**
   ```javascript
   import { Web5 } from '@web5/api';
   
   // Connect to Web5
   const { web5, did } = await Web5.connect();
   console.log('Connected with DID:', did);
   ```

2. **Create and train an AI model**
   ```python
   # ai/train.py
   import torch
   import torch.nn as nn
   
   class SimpleModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.layer = nn.Linear(10, 2)
       
       def forward(self, x):
           return self.layer(x)
   
   # Save model to IPFS
   def save_to_ipfs(model):
       # Implementation for IPFS storage
       pass
   ```

3. **Deploy a smart contract**
   ```solidity
   // contracts/AIModel.sol
   // SPDX-License-Identifier: MIT
   pragma solidity ^0.8.20;
   
   contract AIModel {
       struct Model {
           address owner;
           string ipfsHash;
           uint256 price;
       }
       
       mapping(uint256 => Model) public models;
       uint256 public modelCount;
       
       event ModelTrained(uint256 id, address owner, string ipfsHash);
       
       function trainModel(string memory _ipfsHash) public {
           modelCount++;
           models[modelCount] = Model({
               owner: msg.sender,
               ipfsHash: _ipfsHash,
               price: 0
           });
           
           emit ModelTrained(modelCount, msg.sender, _ipfsHash);
       }
   }
   ```

4. **Connect everything together**
   ```javascript
   // frontend/src/App.jsx
   import { useState } from 'react';
   import { useWeb5 } from './hooks/useWeb5';
   import { trainModel } from './ai/training';
   
   function App() {
     const { web5, did } = useWeb5();
     const [trainingData, setTrainingData] = useState(null);
     
     const handleTrain = async () => {
       // 1. Train model (Web4)
       const model = await trainModel(trainingData);
       
       // 2. Store model in DWN (Web5)
       const { record } = await web5.dwn.records.create({
         data: model,
         message: {
           dataFormat: 'application/octet-stream',
           schema: 'https://schema.org/AIModel'
         }
       });
       
       // 3. Register model on blockchain (Web3)
       const tx = await contract.trainModel(record.id);
       await tx.wait();
       
       console.log('Model trained and registered!');
     };
     
     return (
       <div>
         <h1>Decentralized AI Platform</h1>
         <button onClick={handleTrain}>Train Model</button>
       </div>
     );
   }
   ```

## Testing

Run the test suite:

```bash
# Run smart contract tests
npx hardhat test

# Run AI model tests
pytest ai/tests/

# Run frontend tests
npm test
```

## Deployment

### 1. Smart Contracts
```bash
npx hardhat run scripts/deploy.js --network polygon
```

### 2. Frontend
```bash
npm run build
# Deploy to IPFS/Arweave
```

### 3. AI Models
```bash
# Package and deploy models
python scripts/deploy_model.py
```

## Troubleshooting

### Common Issues

1. **Web5 Connection Failed**
   - Ensure you're using a supported browser
   - Check console for errors
   
2. **Contract Deployment Fails**
   - Verify your .env configuration
   - Ensure you have enough MATIC for gas fees

3. **AI Training Issues**
   - Check CUDA/cuDNN installation for GPU acceleration
   - Verify training data format

## Next Steps

1. Explore the [API Reference](./api-reference.md)
2. Learn about [Smart Contract Development](./smart-contracts.md)
3. Read about [AI Model Training](./ai-training.md)
4. Join our [Community Forum](https://community.aiplatform.org)

## Support

For help, please open an issue or join our [Discord server](https://discord.gg/aiplatform).
