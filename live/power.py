"""Pedido ao Windows para não suspender enquanto um processo roda.

Neste notebook (Modern Standby), a tela apagar por inatividade já põe o sistema em modo de espera, e
alguns minutos depois a rede é cortada. Foi o que interrompeu a gravação do TL2 de Interlagos 2026:
tela apagada às 16:10, rede cortada às 16:18. Por isso os processos ao vivo pedem também a tela acesa.

O pedido vale só enquanto o processo roda e acaba sozinho quando ele termina. Fechar a tampa ou
apertar o botão de energia suspende mesmo assim.
"""
import sys

ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002


def keep_awake(display: bool = False) -> bool:
    """Mantém o sistema acordado (e a tela acesa, com display=True). Devolve se o pedido foi aceito."""
    if sys.platform != "win32":
        return False
    import ctypes
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | (ES_DISPLAY_REQUIRED if display else 0)
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
