"""
ICEBURG Blockchain Switch
Smart Contracts, DeFi Integration, NFTs, and Cross-Chain Operations
"""

from .smart_contracts import ICEBURGSmartContracts, BlockchainVerificationSystem
from .defi_integration import ICEBURGDeFi
from .nft_system import ICEBURGNFTs
from .cross_chain import ICEBURGCrossChain
from .governance import ICEBURGGovernance

__all__ = [
    'ICEBURGSmartContracts',
    'BlockchainVerificationSystem',
    'ICEBURGDeFi',
    'ICEBURGNFTs',
    'ICEBURGCrossChain',
    'ICEBURGGovernance'
]

__version__ = "1.0.0"
__status__ = "BLOCKCHAIN_SWITCH_ACTIVE"
