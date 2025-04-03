import os
import sys
import argparse
import numpy as np
from Bio import PDB
from Bio.PDB import *
import warnings
from scipy.spatial.distance import cdist
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
import mdtraj as md
import prody as pd
from modeller import *
from modeller.optimizers import ConjugateGradients
from modeller.automodel import AutoModel
import concurrent.futures
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

class RotamerLibrary:
    def __init__(self):
        """初始化旋转异构体库"""
        self.rotamers = {
            'ALA': [[0]], 
            'ARG': [[-60, 180], [60, 60], [180, 180], [-60, -60]], 
            'ASN': [[-60], [60], [180]],
            'ASP': [[-60], [60], [180]],
            'CYS': [[-60], [60], [180]],
            'GLN': [[-60, 180], [60, 60], [180, 180]],
            'GLU': [[-60, 180], [60, 60], [180, 180]],
            'HIS': [[-60], [60], [180]],
            'ILE': [[-60], [60], [180]],
            'LEU': [[-60, 180], [60, 60], [180, 180]],
            'LYS': [[-60, 180], [60, 60], [180, 180], [-60, -60]],
            'MET': [[-60, 180], [60, 60], [180, 180]],
            'PHE': [[-60], [60], [180]],
            'PRO': [[30]],
            'SER': [[-60], [60], [180]],
            'THR': [[-60], [60], [180]],
            'TRP': [[-60], [60], [180]],
            'TYR': [[-60], [60], [180]],
            'VAL': [[-60], [60], [180]]
        }

    def get_rotamers(self, residue_name):
        """获取指定氨基酸的旋转异构体"""
        return self.rotamers.get(residue_name, [])

class AntibodyStructureOptimizer:
    def __init__(self, template_file, cdr_file, output_file, cdr_range, chain_id='H'):
        """
        初始化抗体结构优化器
        """
        self.template_file = template_file
        self.cdr_file = cdr_file
        self.output_file = output_file
        self.cdr_range = cdr_range
        self.chain_id = chain_id
        
        self.parser = PDB.PDBParser(QUIET=True)
        self.template = self.parser.get_structure('template', template_file)
        self.cdr = self.parser.get_structure('cdr', cdr_file)
        
        self.rama_thresholds = {
            'favored': [-180, 180, -180, 180],
            'allowed': [-180, 180, -180, 180]
        }
        
        self.setup_energy_parameters()

    def setup_energy_parameters(self):
        """设置能量函数参数"""
        self.bond_length = {
            ('N', 'CA'): 1.46,
            ('CA', 'C'): 1.52,
            ('C', 'O'): 1.23,
            ('C', 'N'): 1.33,
        }
        self.bond_angle = {
            ('N', 'CA', 'C'): 111.0,
            ('CA', 'C', 'O'): 120.5,
            ('CA', 'C', 'N'): 116.6,
            ('C', 'N', 'CA'): 121.7
        }
        
        self.vdw_radii = {
            'C': 1.7,
            'N': 1.55,
            'O': 1.52,
            'S': 1.8
        }
def calculate_rama_score(self, phi, psi):
        """计算Ramachandran图得分"""
        def gaussian(x, mu, sigma):
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        
        alpha_helix = gaussian(phi, -57, 30) * gaussian(psi, -47, 30)
        beta_sheet = gaussian(phi, -119, 30) * gaussian(psi, 113, 30)
        
        return max(alpha_helix, beta_sheet)

    def check_clashes(self, structure, threshold=2.0):
        """检查原子碰撞"""
        clashes = []
        atoms = list(structure.get_atoms())
        coords = np.array([atom.get_coord() for atom in atoms])
        
        ns = NeighborSearch(atoms)
        
        for atom1 in atoms:
            neighbors = ns.search(atom1.get_coord(), threshold)
            for atom2 in neighbors:
                if atom1 != atom2:
                    if not self.are_bonded(atom1, atom2):
                        dist = atom1 - atom2
                        vdw_sum = (self.vdw_radii.get(atom1.element, 1.7) + 
                                 self.vdw_radii.get(atom2.element, 1.7))
                        if dist < vdw_sum:
                            clashes.append((atom1, atom2, dist))
        
        return clashes

    def check_pair_clash(self, atom1, atom2):
        """检查两个原子之间是否存在碰撞"""
        if self.are_bonded(atom1, atom2):
            return False
        
        dist = atom1 - atom2
        vdw_sum = (self.vdw_radii.get(atom1.element, 1.7) + 
                  self.vdw_radii.get(atom2.element, 1.7))
        return dist < vdw_sum

    def are_bonded(self, atom1, atom2):
        """检查两个原子是否共价键相连"""
        res1 = atom1.get_parent()
        res2 = atom2.get_parent()
        
        if res1 == res2:
            return True
        
        if (atom1.get_name() == 'C' and atom2.get_name() == 'N' and 
            res1.get_id()[1] == res2.get_id()[1] - 1):
            return True
        
        return False

    def optimize_backbone(self, residues):
        """优化主链构象"""
        def energy_function(angles):
            """定义能量函数"""
            total_energy = 0
            
            for i, res in enumerate(residues[1:-1]):
                phi, psi = angles[i*2:i*2+2]
                self.set_backbone_angles(res, phi, psi)
            
            total_energy += self.calculate_bond_energy(residues)
            total_energy += self.calculate_angle_energy(residues)
            total_energy += self.calculate_rama_energy(residues)
            total_energy += self.calculate_vdw_energy(residues)
            
            return total_energy
        
        initial_angles = []
        for res in residues[1:-1]:
            phi, psi = self.calculate_backbone_angles(res)
            initial_angles.extend([phi, psi])
        
        result = minimize(energy_function, initial_angles, method='L-BFGS-B',
                        bounds=[(-180, 180)] * len(initial_angles))
        
        return result.x

    def calculate_bond_energy(self, residues):
        """计算键长能量"""
        energy = 0
        k_bond = 1000  # 力常数
        
        for res in residues:
            for atom1_name, atom2_name in self.bond_length:
                if atom1_name in res and atom2_name in res:
                    atom1 = res[atom1_name]
                    atom2 = res[atom2_name]
                    dist = np.linalg.norm(atom1.get_coord() - atom2.get_coord())
                    ideal_dist = self.bond_length[(atom1_name, atom2_name)]
                    energy += k_bond * (dist - ideal_dist) ** 2
                    
        return energy

    def calculate_angle_energy(self, residues):
        """计算键角能量"""
        energy = 0
        k_angle = 500  # 力常数
        
        for res in residues:
            for atoms in self.bond_angle:
                if all(atom in res for atom in atoms):
                    angle = self.calculate_angle(
                        res[atoms[0]].get_coord(),
                        res[atoms[1]].get_coord(),
                        res[atoms[2]].get_coord()
                    )
                    ideal_angle = self.bond_angle[atoms]
                    energy += k_angle * (angle - ideal_angle) ** 2
                    
        return energy
def calculate_rama_energy(self, residues):
        """计算Ramachandran能量"""
        energy = 0
        k_rama = 100  # 权重因子
        
        for res in residues[1:-1]:
            phi, psi = self.calculate_backbone_angles(res)
            rama_score = self.calculate_rama_score(phi, psi)
            energy += k_rama * (1 - rama_score)
            
        return energy

    def calculate_vdw_energy(self, residues):
        """计算van der Waals能量"""
        energy = 0
        k_vdw = 100  # 权重因子
        
        atoms = []
        for res in residues:
            atoms.extend(res.get_atoms())
            
        for i, atom1 in enumerate(atoms):
            for atom2 in atoms[i+1:]:
                if not self.are_bonded(atom1, atom2):
                    dist = np.linalg.norm(atom1.get_coord() - atom2.get_coord())
                    vdw_sum = (self.vdw_radii.get(atom1.element, 1.7) + 
                             self.vdw_radii.get(atom2.element, 1.7))
                    
                    if dist < vdw_sum:
                        energy += k_vdw * (vdw_sum - dist) ** 2
                        
        return energy

    def calculate_angle(self, coord1, coord2, coord3):
        """计算三个原子形成的角度"""
        v1 = coord1 - coord2
        v2 = coord3 - coord2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi

    def calculate_backbone_angles(self, residue):
        """计算主链二面角"""
        phi = psi = 0
        
        prev_res = residue.get_prev_residue()
        next_res = residue.get_next_residue()
        
        if prev_res and 'C' in prev_res and 'N' in residue and 'CA' in residue and 'C' in residue:
            phi = calc_dihedral(prev_res['C'].get_vector(),
                              residue['N'].get_vector(),
                              residue['CA'].get_vector(),
                              residue['C'].get_vector())
            
        if next_res and 'N' in residue and 'CA' in residue and 'C' in residue and 'N' in next_res:
            psi = calc_dihedral(residue['N'].get_vector(),
                              residue['CA'].get_vector(),
                              residue['C'].get_vector(),
                              next_res['N'].get_vector())
            
        return np.degrees(phi), np.degrees(psi)

    def set_backbone_angles(self, residue, phi, psi):
        """设置主链二面角"""
        phi = np.radians(phi)
        psi = np.radians(psi)
        
        if 'N' in residue and 'CA' in residue:
            rotation_matrix = rotaxis(phi, Vector(1, 0, 0))
            residue['N'].transform(rotation_matrix, Vector(0, 0, 0))
            
        if 'CA' in residue and 'C' in residue:
            rotation_matrix = rotaxis(psi, Vector(1, 0, 0))
            residue['C'].transform(rotation_matrix, Vector(0, 0, 0))

    def calculate_chi_angles(self, residue):
        """计算残基的χ角"""
        chi_angles = []
        
        chi_atoms = {
            'ARG': [
                ['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD'],
                ['CB', 'CG', 'CD', 'NE'],
                ['CG', 'CD', 'NE', 'CZ']
            ],
            'LYS': [
                ['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD'],
                ['CB', 'CG', 'CD', 'CE'],
                ['CG', 'CD', 'CE', 'NZ']
            ],
            'ASP': [['N', 'CA', 'CB', 'CG']],
            'ASN': [['N', 'CA', 'CB', 'CG']],
            'GLU': [
                ['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD']
            ],
            'GLN': [
                ['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD']
            ],
            'HIS': [['N', 'CA', 'CB', 'CG']],
            'PHE': [['N', 'CA', 'CB', 'CG']],
            'TRP': [['N', 'CA', 'CB', 'CG']],
            'TYR': [['N', 'CA', 'CB', 'CG']],
            'MET': [
                ['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'SD']
            ],
            'SER': [['N', 'CA', 'CB', 'OG']],
            'THR': [['N', 'CA', 'CB', 'OG1']],
            'CYS': [['N', 'CA', 'CB', 'SG']],
            'ILE': [['N', 'CA', 'CB', 'CG1']],
            'LEU': [
                ['N', 'CA', 'CB', 'CG'],
                ['CA', 'CB', 'CG', 'CD1']
            ],
            'VAL': [['N', 'CA', 'CB', 'CG1']]
        }
        
        res_name = residue.get_resname()
        if res_name in chi_atoms:
            for atoms in chi_atoms[res_name]:
                if all(atom_name in residue for atom_name in atoms):
                    vectors = [residue[atom_name].get_vector() for atom_name in atoms]
                    chi = calc_dihedral(*vectors)
                    chi_angles.append(np.degrees(chi))
        
        return chi_angles
def matches_rotamer(self, chi_angles, rotamer, tolerance=30):
        """检查χ角是否匹配标准旋转异构体"""
        if len(chi_angles) != len(rotamer):
            return False
        
        for chi, rot in zip(chi_angles, rotamer):
            if abs(chi - rot) > tolerance and abs(abs(chi - rot) - 360) > tolerance:
                return False
        
        return True

    def optimize_hydrogen_bonds(self):
        """优化氢键网络"""
        def calculate_hbond_energy(donor, acceptor):
            """计算氢键能量"""
            dist = donor - acceptor
            if dist < 2.0 or dist > 3.5:  # 氢键距离范围
                return float('inf')
            
            # 简单的距离依赖能量函数
            return (dist - 2.8) ** 2  # 2.8Å是理想氢键距离
        
        # 获取所有可能形成氢键的原子
        donors = []
        acceptors = []
        
        for residue in self.template.get_residues():
            for atom in residue:
                # 氢键供体
                if atom.get_name() in ['N', 'NH1', 'NH2', 'NE', 'ND1', 'NE2']:
                    donors.append(atom)
                # 氢键受体
                elif atom.get_name() in ['O', 'OD1', 'OD2', 'OE1', 'OE2']:
                    acceptors.append(atom)
        
        # 优化每个可能的氢键
        for donor in donors:
            min_energy = float('inf')
            best_acceptor = None
            
            for acceptor in acceptors:
                # 跳过同一残基内的原子
                if donor.get_parent() == acceptor.get_parent():
                    continue
                    
                energy = calculate_hbond_energy(donor.get_coord(), acceptor.get_coord())
                if energy < min_energy:
                    min_energy = energy
                    best_acceptor = acceptor
            
            if best_acceptor and min_energy < 1.0:  # 能量阈值
                # 微调原子位置以优化氢键
                donor_residue = donor.get_parent()
                acceptor_residue = best_acceptor.get_parent()
                
                # 小幅调整主链二面角以优化氢键位置
                if not self.are_bonded(donor, best_acceptor):
                    phi, psi = self.calculate_backbone_angles(donor_residue)
                    new_angles = minimize(
                        lambda x: calculate_hbond_energy(
                            donor.get_coord(), 
                            best_acceptor.get_coord()
                        ),
                        [phi, psi],
                        method='L-BFGS-B',
                        bounds=[(-180, 180), (-180, 180)]
                    ).x
                    self.set_backbone_angles(donor_residue, new_angles[0], new_angles[1])

    def elastic_network_adjustment(self, residues, k_elastic=10.0, max_iterations=100):
        """使用弹性网络模型进行局部构象调整"""
        atoms = []
        for res in residues:
            atoms.extend(res.get_atoms())
        
        # 构建初始距离矩阵
        coords = np.array([atom.get_coord() for atom in atoms])
        initial_distances = cdist(coords, coords)
        
        for _ in range(max_iterations):
            forces = np.zeros_like(coords)
            
            # 计算弹性力
            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms)):
                    if not self.are_bonded(atoms[i], atoms[j]):
                        vector = coords[j] - coords[i]
                        distance = np.linalg.norm(vector)
                        if distance < 2.0:  # 碰撞距离阈值
                            force = k_elastic * (2.0 - distance) * vector / distance
                            forces[i] -= force
                            forces[j] += force
            
            # 更新坐标
            coords += forces * 0.1  # 步长因子
            
            # 检查收敛
            if np.all(np.abs(forces) < 0.01):
                break
        
        # 更新原子坐标
        for atom, new_coord in zip(atoms, coords):
            atom.set_coord(new_coord)
def fix_bond_length(self, residue, atom1_name, atom2_name, current_dist):
        """修复键长问题"""
        atom1 = residue[atom1_name]
        atom2 = residue[atom2_name]
        ideal_dist = self.bond_length[(atom1_name, atom2_name)]
        
        # 计算需要移动的距离
        diff = ideal_dist - current_dist
        direction = atom2.get_coord() - atom1.get_coord()
        direction = direction / np.linalg.norm(direction)
        
        # 两个原子各移动一半的距离
        atom1.set_coord(atom1.get_coord() - direction * diff * 0.5)
        atom2.set_coord(atom2.get_coord() + direction * diff * 0.5)

    def fix_bond_angle(self, residue, atoms, current_angle):
        """修复键角问题"""
        atom1 = residue[atoms[0]]
        atom2 = residue[atoms[1]]  # 中心原子
        atom3 = residue[atoms[2]]
        
        ideal_angle = self.bond_angle[atoms]
        
        # 计算需要旋转的角度
        angle_diff = ideal_angle - current_angle
        
        # 创建旋转矩阵
        rotation_axis = np.cross(
            atom1.get_coord() - atom2.get_coord(),
            atom3.get_coord() - atom2.get_coord()
        )
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_matrix = rotaxis(np.radians(angle_diff/2), Vector(rotation_axis))
        
        # 旋转原子
        atom1.transform(rotation_matrix, atom2.get_coord())
        atom3.transform(rotaxis(np.radians(-angle_diff/2), Vector(rotation_axis)), 
                       atom2.get_coord())

    def resolve_clashes(self, clashes):
        """解决原子碰撞"""
        for atom1, atom2, dist in clashes:
            res1 = atom1.get_parent()
            res2 = atom2.get_parent()
            
            # 如果是主链原子碰撞
            if atom1.get_name() in ['N', 'CA', 'C', 'O'] or atom2.get_name() in ['N', 'CA', 'C', 'O']:
                # 微调主链构象
                surrounding_residues = [res1, res2]
                self.optimize_backbone(surrounding_residues)
            else:
                # 侧链碰撞，尝试不同的旋转异构体
                self.optimize_sidechain(res1)
                self.optimize_sidechain(res2)
            
            # 如果仍然存在碰撞，使用弹性网络模型进行局部调整
            if self.check_pair_clash(atom1, atom2):
                self.elastic_network_adjustment([res1, res2])

    def optimize_sidechain(self, residue):
        """优化侧链构象"""
        rotamer_lib = RotamerLibrary()
        best_energy = float('inf')
        best_coords = None
        
        # 保存原始坐标
        original_coords = {}
        for atom in residue:
            original_coords[atom.get_name()] = atom.get_coord()
        
        # 尝试所有可能的旋转异构体
        for rotamer in rotamer_lib.get_rotamers(residue.get_resname()):
            # 应用旋转异构体
            self.apply_rotamer(residue, rotamer)
            
            # 计算能量
            energy = self.calculate_vdw_energy([residue])
            
            if energy < best_energy:
                best_energy = energy
                best_coords = {atom.get_name(): atom.get_coord() for atom in residue}
            
            # 恢复原始坐标以便下次尝试
            for atom in residue:
                atom.set_coord(original_coords[atom.get_name()])
        
        # 应用最佳构象
        if best_coords:
            for atom in residue:
                if atom.get_name() in best_coords:
                    atom.set_coord(best_coords[atom.get_name()])

    def apply_rotamer(self, residue, rotamer):
        """应用旋转异构体构象"""
        residue_name = residue.get_resname()
        
        # 获取侧链原子
        sidechain_atoms = []
        for atom in residue:
            if atom.get_name() not in ['N', 'CA', 'C', 'O']:
                sidechain_atoms.append(atom)
        
        # 应用旋转角度
        for i, chi_angle in enumerate(rotamer):
            if i >= len(sidechain_atoms) - 1:
                break
                
            # 创建旋转矩阵
            rotation_matrix = rotaxis(np.radians(chi_angle), 
                                   Vector(sidechain_atoms[i].get_coord()))
            
            # 对后续原子进行旋转
            for atom in sidechain_atoms[i+1:]:
                atom.transform(rotation_matrix, Vector(0,0,0))
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Antibody Structure Optimization')
    parser.add_argument('--template', required=True, help='Template PDB file')
    parser.add_argument('--cdr', required=True, help='CDR PDB file')
    parser.add_argument('--output', required=True, help='Output PDB file')
    parser.add_argument('--cdr-range', required=True, help='CDR residue range (e.g., 95-102)')
    parser.add_argument('--chain', default='H', help='Chain ID (default: H)')
    
    args = parser.parse_args()
    
    # 解析CDR范围
    start, end = map(int, args.cdr_range.split('-'))
    cdr_range = range(start, end + 1)
    
    # 初始化优化器
    optimizer = AntibodyStructureOptimizer(
        args.template,
        args.cdr,
        args.output,
        cdr_range,
        args.chain
    )
    
    # 获取需要优化的残基
    residues = []
    for model in optimizer.template:
        for chain in model:
            if chain.id == args.chain:
                for residue in chain:
                    if residue.get_id()[1] in cdr_range:
                        residues.append(residue)
    
    # 执行优化
    try:
        # 优化主链构象
        print("Optimizing backbone conformation...")
        optimizer.optimize_backbone(residues)
        
        # 优化侧链构象
        print("Optimizing sidechain conformations...")
        for residue in residues:
            optimizer.optimize_sidechain(residue)
        
        # 检查并解决碰撞
        print("Checking and resolving clashes...")
        clashes = optimizer.check_clashes(optimizer.template)
        if clashes:
            print(f"Found {len(clashes)} clashes, attempting to resolve...")
            optimizer.resolve_clashes(clashes)
        
        # 优化氢键网络
        print("Optimizing hydrogen bond network...")
        optimizer.optimize_hydrogen_bonds()
        
        # 最终的弹性网络调整
        print("Performing final elastic network adjustment...")
        optimizer.elastic_network_adjustment(residues)
        
        # 保存结果
        io = PDBIO()
        io.set_structure(optimizer.template)
        io.save(args.output)
        print(f"Optimization completed. Results saved to {args.output}")
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()