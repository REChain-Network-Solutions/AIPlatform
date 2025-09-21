# Smart Contract Development Guide

This guide provides a comprehensive overview of developing, testing, and deploying smart contracts for the AIPlatform ecosystem.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Development Environment](#development-environment)
- [Writing Smart Contracts](#writing-smart-contracts)
- [Testing](#testing)
- [Deployment](#deployment)
- [Verification](#verification)
- [Security Best Practices](#security-best-practices)
- [Gas Optimization](#gas-optimization)
- [Upgradeable Contracts](#upgradeable-contracts)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Node.js 18+
- npm 9+
- Hardhat or Foundry
- Solidity 0.8.20+
- Git

## Project Structure

```
contracts/
├── governance/     # DAO and voting contracts
│   ├── AIToken.sol
│   ├── Governor.sol
│   └── Treasury.sol
├── marketplace/    # Data and model marketplace
│   ├── DataMarket.sol
│   └── ModelMarket.sol
├── tokens/         # Token contracts
│   ├── ERC20/
│   └── ERC721/
├── interfaces/     # Interfaces
├── libraries/      # Utility libraries
└── test/           # Test files
```

## Development Environment

### 1. Install Dependencies

```bash
npm install --save-dev hardhat @nomicfoundation/hardhat-toolbox @openzeppelin/contracts
```

### 2. Initialize Hardhat

```bash
npx hardhat init
```

### 3. Configure Hardhat

Edit `hardhat.config.js`:

```javascript
require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

module.exports = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  networks: {
    hardhat: {
      chainId: 31337,
    },
    localhost: {
      url: "http://127.0.0.1:8545",
    },
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL || "",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
    },
  },
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY,
  },
};
```

## Writing Smart Contracts

### Example: AI Model Marketplace

```solidity
// contracts/marketplace/ModelMarket.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract ModelMarket is Ownable {
    struct Model {
        address owner;
        string ipfsHash;
        uint256 price;
        bool isActive;
    }

    IERC20 public paymentToken;
    uint256 public platformFee = 2; // 2%
    address public feeRecipient;
    
    mapping(uint256 => Model) public models;
    uint256 public modelCount;

    event ModelListed(uint256 indexed modelId, address indexed owner, string ipfsHash, uint256 price);
    event ModelPurchased(uint256 indexed modelId, address buyer, uint256 price);

    constructor(address _paymentToken, address _feeRecipient) Ownable(msg.sender) {
        paymentToken = IERC20(_paymentToken);
        feeRecipient = _feeRecipient;
    }

    function listModel(string memory _ipfsHash, uint256 _price) external {
        require(bytes(_ipfsHash).length > 0, "Invalid IPFS hash");
        require(_price > 0, "Price must be greater than 0");
        
        modelCount++;
        models[modelCount] = Model({
            owner: msg.sender,
            ipfsHash: _ipfsHash,
            price: _price,
            isActive: true
        });
        
        emit ModelListed(modelCount, msg.sender, _ipfsHash, _price);
    }

    function purchaseModel(uint256 _modelId) external {
        Model storage model = models[_modelId];
        require(model.isActive, "Model not available");
        require(model.owner != msg.sender, "Cannot purchase your own model");
        
        uint256 fee = (model.price * platformFee) / 100;
        uint256 ownerAmount = model.price - fee;
        
        // Transfer payment
        require(
            paymentToken.transferFrom(msg.sender, model.owner, ownerAmount),
            "Payment transfer failed"
        );
        
        if (fee > 0) {
            require(
                paymentToken.transferFrom(msg.sender, feeRecipient, fee),
                "Fee transfer failed"
            );
        }
        
        emit ModelPurchased(_modelId, msg.sender, model.price);
        
        // Transfer ownership (for ERC721, you'd use safeTransferFrom)
        model.owner = msg.sender;
        model.isActive = false;
    }
    
    function setPlatformFee(uint256 _fee) external onlyOwner {
        require(_fee <= 10, "Fee too high");
        platformFee = _fee;
    }
    
    function setFeeRecipient(address _recipient) external onlyOwner {
        require(_recipient != address(0), "Invalid address");
        feeRecipient = _recipient;
    }
}
```

## Testing

### Writing Tests with Hardhat

```javascript
// test/ModelMarket.test.js
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ModelMarket", function () {
  let ModelMarket, modelMarket, paymentToken, owner, seller, buyer;
  const PRICE = ethers.parseEther("1.0");
  const PLATFORM_FEE = 2; // 2%
  
  beforeEach(async () => {
    [owner, seller, buyer] = await ethers.getSigners();
    
    // Deploy mock token
    const Token = await ethers.getContractFactory("MockToken");
    paymentToken = await Token.deploy();
    await paymentToken.waitForDeployment();
    
    // Deploy ModelMarket
    const ModelMarket = await ethers.getContractFactory("ModelMarket");
    modelMarket = await ModelMarket.deploy(
      await paymentToken.getAddress(),
      owner.address
    );
    await modelMarket.waitForDeployment();
    
    // Mint tokens to buyer
    await paymentToken.mint(buyer.address, ethers.parseEther("1000"));
    await paymentToken.connect(buyer).approve(
      await modelMarket.getAddress(),
      ethers.MaxUint256
    );
  });
  
  it("should allow listing a model", async () => {
    await modelMarket.connect(seller).listModel("QmHash123", PRICE);
    const model = await modelMarket.models(1);
    expect(model.owner).to.equal(seller.address);
    expect(model.ipfsHash).to.equal("QmHash123");
    expect(model.price).to.equal(PRICE);
  });
  
  it("should allow purchasing a model", async () => {
    // List a model
    await modelMarket.connect(seller).listModel("QmHash123", PRICE);
    
    // Purchase the model
    await expect(modelMarket.connect(buyer).purchaseModel(1))
      .to.emit(modelMarket, "ModelPurchased")
      .withArgs(1, buyer.address, PRICE);
    
    // Verify ownership transfer and payment
    const model = await modelMarket.models(1);
    expect(model.owner).to.equal(buyer.address);
    expect(model.isActive).to.be.false;
    
    // Verify payment distribution
    const feeAmount = (PRICE * BigInt(PLATFORM_FEE)) / 100n;
    const sellerAmount = PRICE - feeAmount;
    
    expect(await paymentToken.balanceOf(seller.address)).to.equal(sellerAmount);
    expect(await paymentToken.balanceOf(owner.address)).to.equal(feeAmount);
  });
});
```

### Running Tests

```bash
# Run all tests
npx hardhat test

# Run specific test file
npx hardhat test test/ModelMarket.test.js

# Run with gas reporter
REPORT_GAS=true npx hardhat test

# Run with coverage
npx hardhat coverage
```

## Deployment

### 1. Write Deployment Script

```javascript
// scripts/deploy.js
const hre = require("hardhat");

async function main() {
  // Deploy payment token (if needed)
  const Token = await hre.ethers.getContractFactory("MockToken");
  const token = await Token.deploy();
  await token.waitForDeployment();
  console.log(`Token deployed to: ${await token.getAddress()}`);
  
  // Deploy ModelMarket
  const ModelMarket = await hre.ethers.getContractFactory("ModelMarket");
  const modelMarket = await ModelMarket.deploy(
    await token.getAddress(),
    process.env.FEE_RECIPIENT || (await hre.ethers.provider.getSigner(0)).address
  );
  
  await modelMarket.waitForDeployment();
  console.log(`ModelMarket deployed to: ${await modelMarket.getAddress()}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

### 2. Deploy to Network

```bash
# Deploy to localhost
npx hardhat run scripts/deploy.js --network localhost

# Deploy to testnet
npx hardhat run scripts/deploy.js --network sepolia

# Verify on Etherscan
npx hardhat verify --network sepolia DEPLOYED_CONTRACT_ADDRESS "Constructor Argument 1"
```

## Verification

### 1. Flatten Contract (if needed)

```bash
npx hardhat flatten contracts/ModelMarket.sol > ModelMarketFlattened.sol
```

### 2. Verify on Etherscan

```bash
npx hardhat verify --network sepolia \
  --constructor-args arguments.js \
  DEPLOYED_CONTRACT_ADDRESS
```

## Security Best Practices

### 1. Use OpenZeppelin Contracts
```solidity
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
```

### 2. Use SafeMath (pre-0.8.0) or Solidity 0.8.0+
```solidity
// SafeMath is built into Solidity 0.8.0+
uint256 sum = a + b; // Automatically checks for overflow
```

### 3. Use Checks-Effects-Interactions Pattern
```solidity
function withdraw() public {
    uint256 amount = balances[msg.sender];
    
    // 1. Checks
    require(amount > 0, "No balance to withdraw");
    
    // 2. Effects
    balances[msg.sender] = 0;
    
    // 3. Interactions (do last)
    (bool success, ) = msg.sender.call{value: amount}("");
    require(success, "Transfer failed");
}
```

### 4. Use ReentrancyGuard
```solidity
contract MyContract is ReentrancyGuard {
    function withdraw() public nonReentrant {
        // Withdrawal logic
    }
}
```

## Gas Optimization

### 1. Use Constants
```solidity
uint256 public constant MAX_SUPPLY = 10000;
```

### 2. Pack Variables
```solidity
// Before (uses 2 storage slots)
uint128 public a;
uint128 public b;

// After (uses 1 storage slot)
uint128 public a;
uint128 public b;
```

### 3. Use Events for Off-chain Data
```solidity
event UserRegistered(address indexed user, string name, uint256 timestamp);

function register(string memory name) public {
    // ...
    emit UserRegistered(msg.sender, name, block.timestamp);
}
```

## Upgradeable Contracts

### 1. Use OpenZeppelin Upgrades
```solidity
// contracts/MyContract.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";

contract MyContract is Initializable, OwnableUpgradeable {
    uint256 public value;
    
    function initialize(uint256 _value) public initializer {
        __Ownable_init(msg.sender);
        value = _value;
    }
    
    function setValue(uint256 _value) public onlyOwner {
        value = _value;
    }
}
```

### 2. Deploy with Hardhat Upgrades Plugin
```javascript
// scripts/deploy-upgradeable.js
const { ethers, upgrades } = require("hardhat");

async function main() {
  const MyContract = await ethers.getContractFactory("MyContract");
  const contract = await upgrades.deployProxy(MyContract, [42], { initializer: 'initialize' });
  await contract.waitForDeployment();
  console.log("Contract deployed to:", await contract.getAddress());
}

main();
```

## Troubleshooting

### Common Issues

#### 1. Out of Gas
- Increase gas limit in your transaction
- Optimize contract code
- Use view/pure functions for read-only operations

#### 2. Nonce Too Low
- Wait for previous transactions to be mined
- Reset your local node if testing locally

#### 3. Contract Verification Fails
- Ensure constructor arguments are correct
- Verify compiler version and optimization settings
- Try flattening the contract

### Debugging

```bash
# Run Hardhat in debug mode
npx hardhat node --verbose

# Use console.log in your contracts
import "hardhat/console.sol";

function myFunction() public view {
    console.log("Value:", myValue);
}
```

## Additional Resources

- [Solidity Documentation](https://docs.soliditylang.org/)
- [OpenZeppelin Contracts](https://docs.openzeppelin.com/contracts/)
- [Hardhat Documentation](https://hardhat.org/docs/)
- [Ethereum Smart Contract Best Practices](https://consensys.github.io/smart-contract-best-practices/)
- [Smart Contract Security Verification Standard](https://github.com/ethereum-lists/4bytes)

## Getting Help

If you encounter any issues or have questions:

1. Check the [GitHub Issues](https://github.com/REChain-Network-Solutions/AIPlatform/issues)
2. Join our [Discord](https://discord.gg/aiplatform)
3. Ask on [Ethereum Stack Exchange](https://ethereum.stackexchange.com/)
