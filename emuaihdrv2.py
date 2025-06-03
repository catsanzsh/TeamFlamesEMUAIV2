import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import struct
import numpy as np
import os
import time

# N64 Memory Map Constants - A glimpse into the N64's dark heart!
# These are crucial for a ROM to actually find its way around the system.
# We're adding more specific peripheral ranges.
MI_BASE_REG       = 0xA4300000 # MIPS Interface
SP_BASE_REG       = 0xA4000000 # RSP (Reality Signal Processor)
DP_BASE_REG       = 0xA4100000 # RDP (Reality Display Processor)
VI_BASE_REG       = 0xA4400000 # VI (Video Interface)
AI_BASE_REG       = 0xA4500000 # AI (Audio Interface)
PI_BASE_REG       = 0xA4600000 # PI (Parallel Interface - Cartridge/DD access)
RI_BASE_REG       = 0xA4700000 # RI (RDRAM Interface)
SI_BASE_REG       = 0xA4800000 # SI (Serial Interface - Controller access)

# VI (Video Interface) Registers - Where the magic of graphics truly happens!
VI_STATUS_REG     = VI_BASE_REG + 0x00
VI_ORIGIN_REG     = VI_BASE_REG + 0x04 # Framebuffer address in RDRAM
VI_WIDTH_REG      = VI_BASE_REG + 0x08
VI_V_SYNC_REG     = VI_BASE_REG + 0x0C
VI_H_SYNC_REG     = VI_BASE_REG + 0x10
VI_LEAP_REG       = VI_BASE_REG + 0x14
VI_H_START_REG    = VI_BASE_REG + 0x18
VI_V_START_REG    = VI_BASE_REG + 0x1C
VI_V_BURST_REG    = VI_BASE_REG + 0x20
VI_X_SCALE_REG    = VI_BASE_REG + 0x24
VI_Y_SCALE_REG    = VI_BASE_REG + 0x28

class N64Memory:
    def __init__(self):
        self.rdram = bytearray(8 * 1024 * 1024)  # 8MB RDRAM (expandable)
        self.rom = None
        self.sram = bytearray(256 * 1024)  # 256KB SRAM - not fully emulated yet
        self.registers = {} # General purpose registers, though many N64 peripherals have dedicated registers

        # Initialize VI registers with some typical default values
        self.vi_regs = {
            VI_STATUS_REG: 0,
            VI_ORIGIN_REG: 0, # Default framebuffer to 0 (blackness)
            VI_WIDTH_REG: 320,
            VI_V_SYNC_REG: 0,
            VI_H_SYNC_REG: 0,
            VI_LEAP_REG: 0,
            VI_H_START_REG: 0,
            VI_V_START_REG: 0,
            VI_V_BURST_REG: 0,
            VI_X_SCALE_REG: 0x200, # Default scale for 320x240
            VI_Y_SCALE_REG: 0x200,
        }
        
    def load_rom(self, rom_data):
        self.rom = rom_data
        
    def read32(self, address):
        # Memory mapping - now with more delightful complexity!
        # Remember, unmapped reads return 0 or cause exceptions on real hardware.
        # We'll just return 0 for now.
        if 0x00000000 <= address < 0x00800000:  # RDRAM (8MB)
            idx = address & 0x7FFFFF # Mask to stay within 8MB
            if idx + 4 <= len(self.rdram):
                return struct.unpack('>I', self.rdram[idx:idx+4])[0]
            else:
                # Handle out-of-bounds RDRAM access gracefully
                print(f"Warning: RDRAM read out of bounds at 0x{address:08X}")
                return 0
        elif 0x10000000 <= address < 0x1FC00000:  # ROM (Cartridge Domain 1/2)
            if self.rom:
                idx = address - 0x10000000
                if idx + 4 <= len(self.rom):
                    return struct.unpack('>I', self.rom[idx:idx+4])[0]
                else:
                    # Handle out-of-bounds ROM access
                    print(f"Warning: ROM read out of bounds at 0x{address:08X}")
                    return 0
            return 0 # No ROM loaded
        elif VI_BASE_REG <= address < VI_BASE_REG + 0x30: # VI Registers
            if address in self.vi_regs:
                return self.vi_regs[address]
            else:
                print(f"Warning: Unmapped VI register read at 0x{address:08X}")
                return 0
        elif MI_BASE_REG <= address < MI_BASE_REG + 0x20: # MI Registers (dummy)
            return 0 # Placeholder for MIPS Interface registers
        elif SP_BASE_REG <= address < SP_BASE_REG + 0x1000: # RSP Registers (dummy)
            return 0
        elif DP_BASE_REG <= address < DP_BASE_REG + 0x100: # RDP Registers (dummy)
            return 0
        elif AI_BASE_REG <= address < AI_BASE_REG + 0x20: # AI Registers (dummy)
            return 0
        elif PI_BASE_REG <= address < PI_BASE_REG + 0x20: # PI Registers (dummy)
            return 0
        elif RI_BASE_REG <= address < RI_BASE_REG + 0x20: # RI Registers (dummy)
            return 0
        elif SI_BASE_REG <= address < SI_BASE_REG + 0x20: # SI Registers (dummy)
            return 0
        else:
            # Unmapped memory region - a dark abyss!
            # print(f"Warning: Read from unmapped address 0x{address:08X}")
            return 0
        
    def write32(self, address, value):
        if 0x00000000 <= address < 0x00800000:  # RDRAM
            idx = address & 0x7FFFFF
            if idx + 4 <= len(self.rdram):
                struct.pack_into('>I', self.rdram, idx, value)
            else:
                print(f"Warning: RDRAM write out of bounds at 0x{address:08X} value 0x{value:08X}")
        elif VI_BASE_REG <= address < VI_BASE_REG + 0x30: # VI Registers
            if address in self.vi_regs:
                self.vi_regs[address] = value
                # print(f"VI Write: 0x{address:08X} = 0x{value:08X}") # For debugging VI writes
            else:
                print(f"Warning: Unmapped VI register write at 0x{address:08X} value 0x{value:08X}")
        elif MI_BASE_REG <= address < MI_BASE_REG + 0x20: # MI Registers (dummy)
            pass
        elif SP_BASE_REG <= address < SP_BASE_REG + 0x1000: # RSP Registers (dummy)
            pass
        elif DP_BASE_REG <= address < DP_BASE_REG + 0x100: # RDP Registers (dummy)
            pass
        elif AI_BASE_REG <= address < AI_BASE_REG + 0x20: # AI Registers (dummy)
            pass
        elif PI_BASE_REG <= address < PI_BASE_REG + 0x20: # PI Registers (dummy)
            pass
        elif RI_BASE_REG <= address < RI_BASE_REG + 0x20: # RI Registers (dummy)
            pass
        elif SI_BASE_REG <= address < SI_BASE_REG + 0x20: # SI Registers (dummy)
            pass
        else:
            # print(f"Warning: Write to unmapped address 0x{address:08X} value 0x{value:08X}")
            pass

class MIPSR4300i:
    """MIPS R4300i CPU emulation - Now with a few more tricks up its sleeve!"""
    def __init__(self, memory):
        self.memory = memory
        self.pc = 0xA4000040  # Boot vector - The initial spark of life!
        self.regs = [0] * 32  # General purpose registers ($zero is $0)
        self.regs[0] = 0  # $zero always 0
        self.hi = 0       # HI register for multiply/divide
        self.lo = 0       # LO register for multiply/divide
        self.cp0_regs = [0] * 32  # Coprocessor 0 registers (System Control Coprocessor)
        self.cycles = 0
        
    def fetch(self):
        return self.memory.read32(self.pc)
        
    def execute(self):
        instruction = self.fetch()
        opcode = (instruction >> 26) & 0x3F # Our main decision maker!
        
        # Extract common fields for R-type and I-type instructions
        rs = (instruction >> 21) & 0x1F
        rt = (instruction >> 16) & 0x1F
        rd = (instruction >> 11) & 0x1F
        sa = (instruction >> 6) & 0x1F # Shift amount
        
        # Sign extension helper for immediate values
        def sign_extend_16(value):
            return value | (0xFFFF0000 if (value & 0x8000) else 0)
            
        def sign_extend_26(value):
            return value | (0xFC000000 if (value & 0x02000000) else 0)

        # Simplified instruction execution - Let's give it more commands!
        # This is where the CPU performs its dark ballet of operations.
        
        # R-Type Instructions (opcode 0x00)
        if opcode == 0x00:  
            funct = instruction & 0x3F
            if funct == 0x20:  # ADD $rd, $rs, $rt (add with overflow trap, we'll ignore trap for now)
                self.regs[rd] = (self.regs[rs] + self.regs[rt]) & 0xFFFFFFFF
            elif funct == 0x21: # ADDU $rd, $rs, $rt (add unsigned, no overflow trap)
                self.regs[rd] = (self.regs[rs] + self.regs[rt]) & 0xFFFFFFFF
            elif funct == 0x24: # AND $rd, $rs, $rt
                self.regs[rd] = (self.regs[rs] & self.regs[rt]) & 0xFFFFFFFF
            elif funct == 0x27: # NOR $rd, $rs, $rt
                self.regs[rd] = (~(self.regs[rs] | self.regs[rt])) & 0xFFFFFFFF
            elif funct == 0x25: # OR $rd, $rs, $rt
                self.regs[rd] = (self.regs[rs] | self.regs[rt]) & 0xFFFFFFFF
            elif funct == 0x2A: # SLT $rd, $rs, $rt (Set Less Than)
                self.regs[rd] = 1 if (self.regs[rs] < self.regs[rt]) else 0
            elif funct == 0x2B: # SLTU $rd, $rs, $rt (Set Less Than Unsigned)
                self.regs[rd] = 1 if ((self.regs[rs] & 0xFFFFFFFF) < (self.regs[rt] & 0xFFFFFFFF)) else 0
            elif funct == 0x00:  # SLL $rd, $rt, sa (Shift Left Logical)
                self.regs[rd] = (self.regs[rt] << sa) & 0xFFFFFFFF
            elif funct == 0x02:  # SRL $rd, $rt, sa (Shift Right Logical)
                self.regs[rd] = (self.regs[rt] >> sa) & 0xFFFFFFFF # Python's >> handles unsigned logic for positive numbers
            elif funct == 0x03:  # SRA $rd, $rt, sa (Shift Right Arithmetic)
                self.regs[rd] = ((self.regs[rt] ^ 0x80000000) >> sa) ^ 0x80000000 # Proper arithmetic shift
                if self.regs[rt] & 0x80000000: # If negative, fill with ones
                    self.regs[rd] |= (((1 << sa) - 1) << (32 - sa))
            elif funct == 0x04:  # SLLV $rd, $rt, $rs (Shift Left Logical Variable)
                self.regs[rd] = (self.regs[rt] << (self.regs[rs] & 0x1F)) & 0xFFFFFFFF
            elif funct == 0x06:  # SRLV $rd, $rt, $rs (Shift Right Logical Variable)
                self.regs[rd] = (self.regs[rt] >> (self.regs[rs] & 0x1F)) & 0xFFFFFFFF
            elif funct == 0x07:  # SRAV $rd, $rt, $rs (Shift Right Arithmetic Variable)
                self.regs[rd] = ((self.regs[rt] ^ 0x80000000) >> (self.regs[rs] & 0x1F)) ^ 0x80000000
                if self.regs[rt] & 0x80000000:
                    self.regs[rd] |= (((1 << (self.regs[rs] & 0x1F)) - 1) << (32 - (self.regs[rs] & 0x1F)))
            elif funct == 0x08:  # JR $rs (Jump Register)
                self.pc = self.regs[rs] - 4 # Adjust for pc increment
            elif funct == 0x09:  # JALR $rd, $rs (Jump And Link Register)
                self.regs[rd] = self.pc + 8 # Store return address
                self.pc = self.regs[rs] - 4 # Adjust for pc increment
            elif funct == 0x18:  # MULT $rs, $rt (Multiply signed)
                result = (self.regs[rs] * self.regs[rt])
                self.lo = result & 0xFFFFFFFF
                self.hi = (result >> 32) & 0xFFFFFFFF
            elif funct == 0x19:  # MULTU $rs, $rt (Multiply unsigned)
                result = (self.regs[rs] & 0xFFFFFFFF) * (self.regs[rt] & 0xFFFFFFFF)
                self.lo = result & 0xFFFFFFFF
                self.hi = (result >> 32) & 0xFFFFFFFF
            elif funct == 0x10:  # MFHI $rd (Move From HI)
                self.regs[rd] = self.hi
            elif funct == 0x12:  # MFLO $rd (Move From LO)
                self.regs[rd] = self.lo
            elif funct == 0x11:  # MTHI $rs (Move To HI)
                self.hi = self.regs[rs]
            elif funct == 0x13:  # MTLO $rs (Move To LO)
                self.lo = self.regs[rs]
            else:
                # print(f"Warning: Unimplemented R-type funct: 0x{funct:02X} at PC 0x{self.pc:08X}")
                pass
        
        # I-Type Instructions
        elif opcode == 0x08:  # ADDI $rt, $rs, imm (Add Immediate)
            imm = sign_extend_16(instruction & 0xFFFF)
            self.regs[rt] = (self.regs[rs] + imm) & 0xFFFFFFFF
        elif opcode == 0x09:  # ADDIU $rt, $rs, imm (Add Immediate Unsigned)
            imm = sign_extend_16(instruction & 0xFFFF)
            self.regs[rt] = (self.regs[rs] + imm) & 0xFFFFFFFF
        elif opcode == 0x0C:  # ANDI $rt, $rs, imm (AND Immediate)
            imm = instruction & 0xFFFF # Logical AND, so no sign extend
            self.regs[rt] = (self.regs[rs] & imm) & 0xFFFFFFFF
        elif opcode == 0x0D:  # ORI $rt, $rs, imm (OR Immediate)
            imm = instruction & 0xFFFF # Logical OR, so no sign extend
            self.regs[rt] = (self.regs[rs] | imm) & 0xFFFFFFFF
        elif opcode == 0x0E:  # XORI $rt, $rs, imm (XOR Immediate)
            imm = instruction & 0xFFFF
            self.regs[rt] = (self.regs[rs] ^ imm) & 0xFFFFFFFF
        elif opcode == 0x0F:  # LUI $rt, imm (Load Upper Immediate)
            imm = instruction & 0xFFFF
            self.regs[rt] = (imm << 16) & 0xFFFFFFFF
        elif opcode == 0x04:  # BEQ $rs, $rt, offset (Branch Equal)
            offset = sign_extend_16(instruction & 0xFFFF) << 2
            if self.regs[rs] == self.regs[rt]:
                self.pc += offset
        elif opcode == 0x05:  # BNE $rs, $rt, offset (Branch Not Equal)
            offset = sign_extend_16(instruction & 0xFFFF) << 2
            if self.regs[rs] != self.regs[rt]:
                self.pc += offset
        elif opcode == 0x23:  # LW $rt, offset($rs) (Load Word)
            offset = sign_extend_16(instruction & 0xFFFF)
            addr = (self.regs[rs] + offset) & 0xFFFFFFFF
            self.regs[rt] = self.memory.read32(addr)
        elif opcode == 0x2B:  # SW $rt, offset($rs) (Store Word)
            offset = sign_extend_16(instruction & 0xFFFF)
            addr = (self.regs[rs] + offset) & 0xFFFFFFFF
            self.memory.write32(addr, self.regs[rt])
        elif opcode == 0x20:  # LB $rt, offset($rs) (Load Byte)
            offset = sign_extend_16(instruction & 0xFFFF)
            addr = (self.regs[rs] + offset) & 0xFFFFFFFF
            byte_val = self.memory.read32(addr & ~0x3) # Read word containing byte
            byte_val = (byte_val >> ((3 - (addr & 0x3)) * 8)) & 0xFF # Extract correct byte
            self.regs[rt] = sign_extend_16(byte_val) # Sign extend byte to word
        elif opcode == 0x21:  # LH $rt, offset($rs) (Load Halfword)
            offset = sign_extend_16(instruction & 0xFFFF)
            addr = (self.regs[rs] + offset) & 0xFFFFFFFF
            half_val = self.memory.read32(addr & ~0x3) # Read word containing halfword
            half_val = (half_val >> ((2 - ((addr & 0x3) // 2)) * 16)) & 0xFFFF # Extract correct halfword
            self.regs[rt] = sign_extend_16(half_val) # Sign extend halfword to word
        elif opcode == 0x28:  # SB $rt, offset($rs) (Store Byte)
            offset = sign_extend_16(instruction & 0xFFFF)
            addr = (self.regs[rs] + offset) & 0xFFFFFFFF
            current_word = self.memory.read32(addr & ~0x3)
            byte_to_store = self.regs[rt] & 0xFF
            byte_pos_in_word = 3 - (addr & 0x3) # 0 for MSB, 3 for LSB (Big Endian)
            
            # Clear target byte and insert new one
            mask = ~(0xFF << (byte_pos_in_word * 8))
            new_word = (current_word & mask) | (byte_to_store << (byte_pos_in_word * 8))
            self.memory.write32(addr & ~0x3, new_word)
        elif opcode == 0x29:  # SH $rt, offset($rs) (Store Halfword)
            offset = sign_extend_16(instruction & 0xFFFF)
            addr = (self.regs[rs] + offset) & 0xFFFFFFFF
            current_word = self.memory.read32(addr & ~0x3)
            half_to_store = self.regs[rt] & 0xFFFF
            half_pos_in_word = 1 - ((addr & 0x3) // 2) # 0 for MS halfword, 1 for LS halfword
            
            # Clear target halfword and insert new one
            mask = ~(0xFFFF << (half_pos_in_word * 16))
            new_word = (current_word & mask) | (half_to_store << (half_pos_in_word * 16))
            self.memory.write32(addr & ~0x3, new_word)
        
        # J-Type Instructions
        elif opcode == 0x02:  # J target (Jump)
            target = (instruction & 0x3FFFFFF) << 2
            self.pc = (self.pc & 0xF0000000) | target - 4 # Adjust for pc increment
        elif opcode == 0x03:  # JAL target (Jump And Link)
            target = (instruction & 0x3FFFFFF) << 2
            self.regs[31] = self.pc + 8 # Store return address in $ra (reg 31)
            self.pc = (self.pc & 0xF0000000) | target - 4 # Adjust for pc increment
        
        # Coprocessor 0 Instructions (for system control)
        elif opcode == 0x10: # COP0 (Coprocessor 0)
            sub_opcode = (rs & 0x1F) # Used for MFC0, MTC0
            rd_cp0 = (instruction >> 11) & 0x1F # For MFC0
            
            if sub_opcode == 0x00: # MFC0 $rt, $rd_cp0 (Move From Coprocessor 0)
                self.regs[rt] = self.cp0_regs[rd_cp0]
            elif sub_opcode == 0x04: # MTC0 $rt, $rd_cp0 (Move To Coprocessor 0)
                self.cp0_regs[rd_cp0] = self.regs[rt]
                # In a real emulator, writing to certain CP0 registers (like Status, Cause)
                # would trigger specific actions like interrupt enable/disable, mode changes.
                # For now, we just store the value.
            else:
                # print(f"Warning: Unimplemented COP0 sub-opcode: 0x{sub_opcode:02X} at PC 0x{self.pc:08X}")
                pass

        else:
            # The ROM is trying to do something we haven't taught our CPU yet!
            # print(f"Warning: Unimplemented opcode: 0x{opcode:02X} at PC 0x{self.pc:08X}")
            pass # We'll just let it slide for now

        self.pc += 4 # Next instruction, unless a branch/jump changed PC
        self.cycles += 1
        self.regs[0] = 0  # $zero always 0 - unwavering!

class RCP:
    """Reality Coprocessor (RSP + RDP) - Now with a peek into the RDRAM framebuffer!"""
    def __init__(self, memory):
        self.memory = memory
        self.framebuffer = np.zeros((240, 320, 3), dtype=np.uint8) # Default black for safety
        
    def render_frame(self):
        # The N64 typically renders to a framebuffer in RDRAM.
        # We need to find the address and dimensions from the VI registers.
        fb_address = self.memory.vi_regs.get(VI_ORIGIN_REG, 0)
        width = self.memory.vi_regs.get(VI_WIDTH_REG, 320)
        
        # Simple rendering for now: try to pull data from RDRAM.
        # N64 supports 16-bit (RGB555/RGBA5551) and 32-bit (RGBA8888) framebuffers.
        # Let's assume a simple 16-bit RGB555 format for a basic visualization.
        # This is a massive simplification, as RDP commands actually draw pixels.
        
        height = 240 # N64 common resolutions (e.g., 320x240, 640x480)
        
        if width == 0: width = 320 # Prevent division by zero if width not set
        
        # This will simulate reading pixel data from the framebuffer address
        # as if it were RGB555. It's a crude representation, but it's *something*
        # from the emulated memory, not just a static color!
        
        # Determine bytes per pixel based on actual VI format (we're guessing 16-bit here)
        # N64 VI_STATUS_REG bits control pixel format:
        # 16-bit (A1R5G5B5/R5G5B5A1) usually at 0x00000002
        # 32-bit (A8R8G8B8/R8G8B8A8) usually at 0x00000003
        
        vi_status = self.memory.vi_regs.get(VI_STATUS_REG, 0)
        if (vi_status & 0x3) == 0x3: # Assume 32-bit RGBA8888 for example
            bytes_per_pixel = 4
            # print("32-bit framebuffer detected (assumption)")
        else: # Assume 16-bit RGB555/RGBA5551
            bytes_per_pixel = 2
            # print("16-bit framebuffer detected (assumption)")

        expected_fb_size = width * height * bytes_per_pixel
        
        # Ensure the framebuffer address is within RDRAM bounds
        if not (0x00000000 <= fb_address < 0x00800000):
            # print(f"Warning: Invalid framebuffer address 0x{fb_address:08X}. Defaulting to black.")
            self.framebuffer.fill(0) # Keep it black if address is bad
            return self.framebuffer

        # Attempt to read the raw bytes from RDRAM
        rdram_offset = fb_address & 0x7FFFFF
        
        # Ensure we don't try to read past the end of RDRAM
        if rdram_offset + expected_fb_size > len(self.memory.rdram):
            # print(f"Warning: Framebuffer exceeds RDRAM bounds. Read truncated or defaulted.")
            data_to_read = self.memory.rdram[rdram_offset:] # Read what's available
            
            # Pad with zeros if we don't have enough data to fill the frame
            # This is a bit of a hack but prevents errors if the FB is too large/misaligned.
            if len(data_to_read) < expected_fb_size:
                data_to_read += bytearray(expected_fb_size - len(data_to_read))
        else:
            data_to_read = self.memory.rdram[rdram_offset : rdram_offset + expected_fb_size]

        # Reshape data into a NumPy array and convert to RGB
        try:
            if bytes_per_pixel == 4: # RGBA8888
                # N64 is Big Endian, so AARRGGBB in memory
                # We need RR GG BB for PIL Image.
                # Assuming data_to_read is already bytearray
                pixels = np.frombuffer(data_to_read, dtype=np.uint32).reshape((height, width))
                
                # Extract R, G, B channels
                r = ((pixels >> 16) & 0xFF).astype(np.uint8)
                g = ((pixels >> 8) & 0xFF).astype(np.uint8)
                b = (pixels & 0xFF).astype(np.uint8)
                
                self.framebuffer = np.stack([r, g, b], axis=-1)

            else: # 16-bit RGB555 / RGBA5551 (common default)
                # N64 is Big Endian. A1R5G5B5 or R5G5B5A1. Let's assume R5G5B5.
                # Data is stored as two bytes per pixel.
                pixels = np.frombuffer(data_to_read, dtype=np.uint16).reshape((height, width))
                
                # Convert 5-bit channels to 8-bit. (Val << 3) | (Val >> 2) for R5G5B5
                # R = (pixel >> 11) & 0x1F  (5 bits)
                # G = (pixel >> 6) & 0x1F   (5 bits)
                # B = (pixel >> 1) & 0x1F   (5 bits)
                
                r_5bit = (pixels >> 11) & 0x1F
                g_5bit = (pixels >> 6) & 0x1F
                b_5bit = (pixels >> 1) & 0x1F
                
                r = (r_5bit << 3 | r_5bit >> 2).astype(np.uint8)
                g = (g_5bit << 3 | g_5bit >> 2).astype(np.uint8)
                b = (b_5bit << 3 | b_5bit >> 2).astype(np.uint8)
                
                self.framebuffer = np.stack([r, g, b], axis=-1)

        except ValueError as e:
            # This can happen if the ROM sets weird resolutions or framebuffer sizes
            # that don't match our assumptions or if RDRAM data is corrupt.
            # print(f"Error reading framebuffer data: {e}. Defaulting to black.")
            self.framebuffer.fill(0)
        except Exception as e:
            # print(f"Unexpected error in rendering frame: {e}. Defaulting to black.")
            self.framebuffer.fill(0)

        return self.framebuffer

class N64Emulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Project64 - Nintendo 64 Emulator (Cat-san's Touch!)")
        self.root.geometry("800x600")
        
        # Set dark theme - for our dark deeds...
        self.root.configure(bg='#2b2b2b')
        
        self.memory = N64Memory()
        self.cpu = MIPSR4300i(self.memory)
        self.rcp = RCP(self.memory) # Pass memory to RCP so it can read VI registers
        
        self.rom_loaded = False
        self.running = False
        self.fps = 0
        self.last_frame_time = time.time()
        
        self.setup_gui()
        
    def setup_gui(self):
        # Menu bar
        menubar = tk.Menu(self.root, bg='#3c3c3c', fg='white')
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg='#3c3c3c', fg='white')
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open ROM...", command=self.load_rom, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # System menu
        system_menu = tk.Menu(menubar, tearoff=0, bg='#3c3c3c', fg='white')
        menubar.add_cascade(label="System", menu=system_menu)
        system_menu.add_command(label="Start", command=self.start_emulation, accelerator="F5")
        system_menu.add_command(label="Pause", command=self.pause_emulation, accelerator="F6")
        system_menu.add_command(label="Reset", command=self.reset_emulation, accelerator="F8")
        
        # Options menu
        options_menu = tk.Menu(menubar, tearoff=0, bg='#3c3c3c', fg='white')
        menubar.add_cascade(label="Options", menu=options_menu)
        options_menu.add_command(label="Configure Graphics...")
        options_menu.add_command(label="Configure Audio...")
        options_menu.add_command(label="Configure Controller...")
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg='#3c3c3c', fg='white')
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Toolbar
        toolbar = tk.Frame(self.root, bg='#3c3c3c', height=40)
        toolbar.pack(fill=tk.X)
        
        # Toolbar buttons
        button_style = {'bg': '#4a4a4a', 'fg': 'white', 'bd': 1, 'relief': tk.RAISED, 'padx': 10}
        
        tk.Button(toolbar, text="📁 Open", command=self.load_rom, **button_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="▶ Start", command=self.start_emulation, **button_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="⏸ Pause", command=self.pause_emulation, **button_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="⏹ Stop", command=self.stop_emulation, **button_style).pack(side=tk.LEFT, padx=2, pady=5)
        tk.Button(toolbar, text="🔄 Reset", command=self.reset_emulation, **button_style).pack(side=tk.LEFT, padx=2, pady=5)
        
        # Main display area
        self.display_frame = tk.Frame(self.root, bg='black')
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for rendering
        self.canvas = tk.Canvas(self.display_frame, width=640, height=480, bg='black', highlightthickness=0)
        self.canvas.pack(expand=True)
        
        # Status bar
        self.status_bar = tk.Frame(self.root, bg='#2b2b2b', height=25)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(self.status_bar, text="Ready", bg='#2b2b2b', fg='white', anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.fps_label = tk.Label(self.status_bar, text="FPS: 0", bg='#2b2b2b', fg='white', anchor=tk.E)
        self.fps_label.pack(side=tk.RIGHT, padx=5)
        
        # Initialize display
        self.image = Image.new("RGB", (320, 240)) # Default starting size
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas_image = self.canvas.create_image(320, 240, image=self.tk_image, anchor=tk.CENTER) # Centered
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.load_rom())
        self.root.bind('<F5>', lambda e: self.start_emulation())
        self.root.bind('<F6>', lambda e: self.pause_emulation())
        self.root.bind('<F8>', lambda e: self.reset_emulation())
        
        # Resize canvas when window resizes
        self.display_frame.bind('<Configure>', self.on_display_frame_resize)

    def on_display_frame_resize(self, event):
        # Resize canvas to fit the frame, maintaining aspect ratio or filling
        canvas_width = event.width
        canvas_height = event.height
        
        # For N64, common aspect ratio is 4:3 (e.g., 320x240, 640x480)
        # We want to scale the N64 output to fit the canvas while maintaining ratio.
        # If the N64 output is 320x240, the target display size is 640x480 (2x scale)
        # Or even higher if the window is larger.

        # Let's say our internal rendering resolution is 320x240 (rcp.framebuffer)
        # We need to calculate how much to scale it to fit the canvas.
        native_width = 320
        native_height = 240

        # Calculate scale factors for width and height
        scale_w = canvas_width / native_width
        scale_h = canvas_height / native_height
        
        # Use the smaller scale factor to maintain aspect ratio and fit within bounds
        scale = min(scale_w, scale_h)
        
        # Calculate new image dimensions
        new_img_width = int(native_width * scale)
        new_img_height = int(native_height * scale)
        
        self.canvas.config(width=new_img_width, height=new_img_height)
        
        # Recenter image on canvas
        self.canvas.coords(self.canvas_image, new_img_width / 2, new_img_height / 2)


    def load_rom(self):
        filename = filedialog.askopenfilename(
            title="Select N64 ROM",
            filetypes=[("N64 ROMs", "*.z64 *.n64 *.v64"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'rb') as f:
                    rom_data = f.read()
                
                # Detect ROM format and endianness - crucial for a ROM to be readable!
                header = rom_data[:4]
                if header == b'\x80\x37\x12\x40':  # .z64 (big endian - native)
                    pass
                elif header == b'\x37\x80\x40\x12':  # .n64 (little endian)
                    rom_data = self.swap_endian(rom_data)
                elif header == b'\x40\x12\x37\x80':  # .v64 (byte swapped)
                    rom_data = self.byte_swap(rom_data)
                else:
                    messagebox.showwarning("Unknown ROM Format", 
                                           f"ROM header 0x{header.hex()} not recognized. Attempting to load as .z64 anyway.")
                    # If unknown, just try loading as big endian, might work for some
                    pass 
                
                self.memory.load_rom(rom_data)
                self.rom_loaded = True
                self.reset_emulation() # Reset state after loading new ROM
                
                # Extract ROM info - give it a proper name!
                rom_name_bytes = rom_data[0x20:0x34]
                try:
                    rom_name = rom_name_bytes.decode('shift_jis', errors='ignore').strip()
                except UnicodeDecodeError:
                    rom_name = rom_name_bytes.decode('ascii', errors='ignore').strip()

                if not rom_name:
                    rom_name = os.path.basename(filename) # Fallback if name is empty
                
                self.status_label.config(text=f"Loaded: {rom_name}")
                self.root.title(f"Project64 - {rom_name} (Cat-san's Touch!)")
                
                # Clear canvas after loading
                self.image = Image.new("RGB", (320, 240)) # Reset image buffer
                self.tk_image = ImageTk.PhotoImage(self.image)
                self.canvas.itemconfig(self.canvas_image, image=self.tk_image)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ROM: {str(e)}\nMake sure it's a valid N64 ROM and not corrupted.")
    
    def swap_endian(self, data):
        # Convert little endian to big endian (byte by byte swap within 32-bit words)
        # e.g., 0x12345678 (LE) -> 0x78563412 (Big Endian as 32-bit word)
        # But for N64, it's more like 0x12345678 (LE) -> 0x80371240 (Big Endian)
        # The common .n64 format means 32-bit words are byte-swapped from .z64.
        # This means 0xAB CD EF GH becomes 0xGH EF CD AB.
        # For N64: .z64 is ABCD, .n64 is DCBA, .v64 is BADC
        result = bytearray(len(data))
        for i in range(0, len(data), 4):
            if i + 4 <= len(data):
                # Bytes (0,1,2,3) -> (3,2,1,0) for word swap
                result[i+0] = data[i+3]
                result[i+1] = data[i+2]
                result[i+2] = data[i+1]
                result[i+3] = data[i+0]
            else: # Handle partial words at the end
                result[i:] = data[i:]
        return bytes(result)
    
    def byte_swap(self, data):
        # Byte swap for .v64 format (16-bit word swap)
        # e.g., 0xAB CD EF GH becomes 0xCD AB GH EF
        result = bytearray(len(data))
        for i in range(0, len(data), 2):
            if i + 1 < len(data): # Ensure we don't go out of bounds
                result[i] = data[i+1]
                result[i+1] = data[i]
            else: # Handle odd byte at the end if any
                result[i] = data[i]
        return bytes(result)
    
    def start_emulation(self):
        if not self.rom_loaded:
            messagebox.showwarning("Warning", "Please load a ROM first, my dear!")
            return
            
        self.running = True
        self.status_label.config(text="Running - CPU purring...")
        self.emulation_loop()
    
    def pause_emulation(self):
        if self.running:
            self.running = False
            self.status_label.config(text="Paused - Naughty, naughty, don't stop the fun!")
    
    def stop_emulation(self):
        self.running = False
        self.reset_emulation()
        self.status_label.config(text="Stopped - Till next time, my little plaything...")
    
    def reset_emulation(self):
        # A fresh start, like a clean slate for new mischief!
        self.cpu = MIPSR4300i(self.memory) # CPU reset, PC to boot vector
        # self.rcp = RCP(self.memory) # RCP doesn't need full reset, just its internal state (which is currently just fb_address)
        # Clear RDRAM to a known state on reset (optional but good for stability)
        self.memory.rdram = bytearray(8 * 1024 * 1024)
        # Reset VI registers to default state (Crucial for graphics initialization)
        self.memory.vi_regs = {
            VI_STATUS_REG: 0,
            VI_ORIGIN_REG: 0,
            VI_WIDTH_REG: 320,
            VI_V_SYNC_REG: 0,
            VI_H_SYNC_REG: 0,
            VI_LEAP_REG: 0,
            VI_H_START_REG: 0,
            VI_V_START_REG: 0,
            VI_V_BURST_REG: 0,
            VI_X_SCALE_REG: 0x200,
            VI_Y_SCALE_REG: 0x200,
        }
        
        if self.rom_loaded:
            self.status_label.config(text="Reset - Ready for more action!")
            # Re-initialize the display to a blank screen after reset
            self.image = Image.new("RGB", (320, 240))
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.itemconfig(self.canvas_image, image=self.tk_image)
    
    def emulation_loop(self):
        if not self.running:
            return
            
        # The heart of the beast! We run CPU cycles and then paint our picture.
        # A real N64 runs at 93.75 MHz for CPU, and RSP is 62.5 MHz.
        # A frame (60 FPS) is ~1,562,500 CPU cycles.
        # We'll do a simplified fixed number of instructions per frame.
        cycles_per_frame = 100000 # Let's try more to get some work done by the CPU
        
        for _ in range(cycles_per_frame):
            self.cpu.execute()
        
        # Render frame from the RCP's dark arts (reading from RDRAM)
        frame = self.rcp.render_frame()
        
        # Update display - resizing it for your viewing pleasure!
        # Always use the dimensions from the rendered frame for PIL
        current_width = frame.shape[1]
        current_height = frame.shape[0]

        self.image = Image.fromarray(frame, 'RGB')
        
        # Resize to fit the canvas while maintaining aspect ratio
        canvas_info = self.canvas.winfo_width(), self.canvas.winfo_height()
        if canvas_info[0] <= 1 or canvas_info[1] <= 1: # If canvas not yet rendered or too small
            scale_width, scale_height = 640, 480 # Default for display (e.g., 2x native)
        else:
            scale_width, scale_height = canvas_info[0], canvas_info[1]

        # Calculate best fit while maintaining aspect ratio
        aspect_ratio = current_width / current_height
        
        if (scale_width / aspect_ratio) <= scale_height:
            display_width = scale_width
            display_height = int(scale_width / aspect_ratio)
        else:
            display_height = scale_height
            display_width = int(scale_height * aspect_ratio)

        self.image = self.image.resize((display_width, display_height), Image.NEAREST) # Pixelated charm!
        
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.canvas_image, image=self.tk_image)
        
        # Recenter the image on the canvas
        self.canvas.coords(self.canvas_image, self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)

        # Calculate FPS - keeping track of our performance!
        current_time = time.time()
        if (current_time - self.last_frame_time) > 0: # Avoid division by zero
            self.fps = 1.0 / (current_time - self.last_frame_time)
        else:
            self.fps = 0 # If time hasn't advanced, assume 0 FPS
        self.last_frame_time = current_time
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        
        # Schedule next frame - the endless dance!
        self.root.after(16, self.emulation_loop)  # Aim for ~60 FPS (1000ms / 60 frames = ~16.67ms per frame)
    
    def show_about(self):
        messagebox.showinfo("About", 
            "Project64 Style N64 Emulator\n\n"
            "Embraced by Cat-san's devious spirit, this Nintendo 64 emulator "
            "is built with Python and Tkinter.\n"
            "Version 1.0.1 (Now with extra *meow*!)\n\n"
            "This is a dramatically simplified educational implementation, "
            "a mere whisper of the N64's true complexity. "
            "But it's a start, isn't it? Perhaps we can make it do more... "
            "unconventional things later. 😉")

if __name__ == "__main__":
    root = tk.Tk()
    emulator = N64Emulator(root)
    root.mainloop()
