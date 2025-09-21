# Tokenomics and Staking Guide

This document outlines the tokenomics model and staking mechanisms for the AIPlatform ecosystem.

## Table of Contents

- [Token Overview](#token-overview)
- [Token Distribution](#token-distribution)
- [Staking Mechanics](#staking-mechanics)
- [Rewards System](#rewards-system)
- [Governance](#governance)
- [Security Considerations](#security-considerations)

## Token Overview

**Token Name**: AI Platform Token (AIPT)  
**Symbol**: AIPT  
**Decimals**: 18  
**Total Supply**: 1,000,000,000 AIPT  
**Blockchain**: EVM-compatible chains  

## Token Distribution

| Category | Percentage | Tokens | Vesting Period |
|----------|------------|--------|----------------|
| Team & Advisors | 15% | 150M AIPT | 3-year linear vesting |
| Foundation | 20% | 200M AIPT | 4-year vesting |
| Ecosystem & Development | 25% | 250M AIPT | 5-year release |
| Community Rewards | 20% | 200M AIPT | Ongoing |
| Public Sale | 10% | 100M AIPT | - |
| Liquidity | 10% | 100M AIPT | 1-year lock |

## Staking Mechanics

### 1. Staking Pools

| Pool | Lockup Period | APY | Minimum Stake |
|------|---------------|-----|---------------|
| Short-term | 30 days | 8% | 1,000 AIPT |
| Medium-term | 90 days | 15% | 5,000 AIPT |
| Long-term | 180 days | 25% | 10,000 AIPT |
| Validator | 365 days | 35% | 100,000 AIPT |

### 2. Staking Contract

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract AIPTStaking is ReentrancyGuard {
    IERC20 public immutable stakingToken;
    
    struct Stake {
        uint256 amount;
        uint256 stakedAt;
        uint256 lockPeriod;
        bool claimed;
    }
    
    mapping(address => Stake[]) public userStakes;
    
    // APY for different lock periods (in basis points, 10000 = 100%)
    mapping(uint256 => uint256) public apyForLockPeriod;
    
    event Staked(address indexed user, uint256 amount, uint256 lockPeriod);
    event Unstaked(address indexed user, uint256 amount, uint256 reward);
    
    constructor(address _stakingToken) {
        stakingToken = IERC20(_stakingToken);
        
        // Initialize APY for different lock periods (in days)
        apyForLockPeriod[30] = 800;   // 8%
        apyForLockPeriod[90] = 1500;  // 15%
        apyForLockPeriod[180] = 2500; // 25%
        apyForLockPeriod[365] = 3500; // 35%
    }
    
    function stake(uint256 amount, uint256 lockPeriodDays) external nonReentrant {
        require(amount > 0, "Cannot stake 0");
        require(apyForLockPeriod[lockPeriodDays] > 0, "Invalid lock period");
        
        // Transfer tokens from user to contract
        bool success = stakingToken.transferFrom(msg.sender, address(this), amount);
        require(success, "Transfer failed");
        
        // Create new stake
        userStakes[msg.sender].push(Stake({
            amount: amount,
            stakedAt: block.timestamp,
            lockPeriod: lockPeriodDays * 1 days,
            claimed: false
        }));
        
        emit Staked(msg.sender, amount, lockPeriodDays);
    }
    
    function unstake(uint256 stakeIndex) external nonReentrant {
        require(stakeIndex < userStakes[msg.sender].length, "Invalid stake index");
        
        Stake storage userStake = userStakes[msg.sender][stakeIndex];
        require(!userStake.claimed, "Already claimed");
        require(
            block.timestamp >= userStake.stakedAt + userStake.lockPeriod,
            "Lock period not ended"
        );
        
        // Calculate reward
        uint256 lockPeriodInYears = userStake.lockPeriod / 365 days;
        uint256 apy = apyForLockPeriod[userStake.lockPeriod / 1 days];
        uint256 reward = (userStake.amount * apy * lockPeriodInYears) / 10000;
        
        // Mark as claimed
        userStake.claimed = true;
        
        // Transfer staked amount + reward
        bool success = stakingToken.transfer(msg.sender, userStake.amount + reward);
        require(success, "Transfer failed");
        
        emit Unstaked(msg.sender, userStake.amount, reward);
    }
    
    function calculateReward(address user, uint256 stakeIndex) public view returns (uint256) {
        if (stakeIndex >= userStakes[user].length) return 0;
        
        Stake memory userStake = userStakes[user][stakeIndex];
        if (userStake.claimed) return 0;
        
        uint256 timeStaked = block.timestamp - userStake.stakedAt;
        uint256 lockPeriodInYears = (userStake.lockPeriod * 1e18) / 365 days;
        uint256 apy = apyForLockPeriod[userStake.lockPeriod / 1 days];
        
        // Calculate reward based on time staked
        uint256 reward = (userStake.amount * apy * timeStaked) / (365 days * 10000);
        
        return reward;
    }
    
    function getUserStakes(address user) external view returns (Stake[] memory) {
        return userStakes[user];
    }
}
```

## Rewards System

### 1. Reward Sources

1. **Transaction Fees**
   - 0.3% of all transactions go to the staking pool
   - Distributed proportionally to stakers

2. **Network Usage**
   - 40% of platform fees distributed to stakers
   - 60% allocated to the foundation

3. **Liquidity Mining**
   - Additional AIPT rewards for providing liquidity
   - Dynamic APY based on TVL and trading volume

### 2. Reward Distribution

```solidity
function distributeFees(uint256 amount) external onlyOwner {
    require(amount > 0, "Amount must be > 0");
    
    // Transfer tokens to this contract
    bool success = stakingToken.transferFrom(msg.sender, address(this), amount);
    require(success, "Transfer failed");
    
    // Update rewards per token
    // Implementation depends on your specific reward distribution logic
}
```

## Governance

### 1. Voting Power

- 1 AIPT = 1 vote
- Staked tokens receive 2x voting power
- Minimum 0.1% of total supply to create proposals

### 2. Proposal Types

1. **Parameter Changes**
   - Adjust staking rewards
   - Update fees
   - Modify quorum requirements

2. **Treasury Management**
   - Allocate funds to development
   - Grant programs
   - Partnerships

3. **Protocol Upgrades**
   - Smart contract upgrades
   - New features
   - Security improvements

## Security Considerations

### 1. Smart Contract Security

- Audited by multiple security firms
- Time locks for critical functions
- Multi-signature wallet for admin controls
- Emergency pause functionality

### 2. Risk Mitigation

- Maximum 20% of total supply can be staked in a single address
- 72-hour timelock for governance parameter changes
- 14-day voting period for major upgrades

### 3. Insurance Fund

- 5% of all transaction fees go to insurance
- Used to cover potential exploits
- Managed by governance

## Integration Guide

### 1. Connect to Staking Contract

```javascript
const stakingABI = [
  // Staking ABI here
];

const stakingAddress = '0x...';
const provider = new ethers.providers.Web3Provider(window.ethereum);
const signer = provider.getSigner();
const stakingContract = new ethers.Contract(stakingAddress, stakingABI, signer);
```

### 2. Stake Tokens

```javascript
async function stake(amount, days) {
  const tx = await stakingContract.stake(amount, days);
  await tx.wait();
  console.log('Staked successfully');
}
```

### 3. Claim Rewards

```javascript
async function claimRewards(stakeIndex) {
  const tx = await stakingContract.unstake(stakeIndex);
  await tx.wait();
  console.log('Rewards claimed');
}
```

## Monitoring and Analytics

### 1. Key Metrics

- Total Value Locked (TVL)
- Annual Percentage Yield (APY)
- Number of Stakers
- Average Stake Duration
- Reward Distribution

### 2. Dashboard Integration

```javascript
async function getStakingStats() {
  const [tvl, apy, stakers] = await Promise.all([
    stakingContract.totalValueLocked(),
    stakingContract.currentAPY(),
    stakingContract.totalStakers()
  ]);
  
  return { tvl, apy, stakers };
}
```

## Troubleshooting

### Common Issues

1. **Transaction Failing**
   - Check gas fees
   - Ensure token approval
   - Verify sufficient balance

2. **Rewards Not Showing**
   - Wait for next reward cycle
   - Check if lock period has ended
   - Verify staking transaction

3. **Cannot Unstake**
   - Check lock period
   - Ensure not already claimed
   - Verify wallet connection

## Support

For additional help, please contact:
- [Documentation](https://docs.aiplatform.com)
- [Discord](https://discord.gg/aiplatform)
- [GitHub Issues](https://github.com/REChain-Network-Solutions/AIPlatform/issues)
