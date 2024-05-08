import random as rd
import numpy as np
import csv

MUTATION_RATE = 40  # Updated mutation rate
MUTATION_COUNT = 2
MAXFITNESS = 700
csvfile = '2024_AI_TSP.csv'

def read_csv(file):
    """Read CSV file and return data as numpy array."""
    return np.genfromtxt(open(file, "rb"), dtype=float, delimiter=",", skip_header=0)

city_list = read_csv(csvfile)
city_size = len(city_list)

def distance(x, y):
    dist = np.linalg.norm(np.array(x) - np.array(y))
    return dist

def cost(visit):
    total_cost = sum(distance(
        [float(city_list[visit[n]][0]), float(city_list[visit[n]][1])],
        [float(city_list[visit[n+1]][0]), float(city_list[visit[n+1]][1])]
    ) for n in range(len(visit)-1))
    return total_cost

                                                      
def tournament_selection(population, k=10):
    selected = rd.sample(population, k)
    return min(selected, key=lambda x: cost(x, city_list))

def mutation(chromosome, dist_matrix):
    """Perform mutation by swapping two random elements in the chromosome."""
    idx1, idx2 = rd.sample(range(1, len(chromosome) - 1), 2)
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

    # 두 도시의 순서를 변경하여 거리를 다시 계산
    new_distance = 0
    for i in range(len(chromosome) - 1):
        new_distance += dist_matrix[chromosome[i]][chromosome[i+1]]

    return chromosome, new_distance

def crossover(parent1, parent2, distance_matrix):
    """단순한 한 점 교차를 수행합니다."""
    cut = rd.randint(1, len(parent1) - 2)
    child1 = parent1[:cut] + [p for p in parent2 if p not in parent1[:cut]]
    child2 = parent2[:cut] + [p for p in parent1 if p not in parent2[:cut]]
    return child1, child2

def create_distance_matrix(city_coordinates):
    """Create distance matrix based on city coordinates."""
    n = len(city_coordinates)
    distance_matrix = [[0] * n for _ in range(n)]  # n x n 크기의 0으로 초기화된 행렬 생성

    # 모든 도시 쌍에 대해 거리 계산
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = distance(city_coordinates[i], city_coordinates[j])
                distance_matrix[i][j] = dist  # 거리 행렬에 거리값 할당

    return distance_matrix

distance_matrix = create_distance_matrix(city_list)


def create_proximity_list(distance_matrix, num_neighbors=10):
    proximity_list = []

    # 거리 행렬에서 각 도시에 대한 근접 도시 목록 생성
    for i in range(len(distance_matrix)):
        distances = distance_matrix[i]
        # 거리와 도시 인덱스를 튜플로 저장하여 리스트에 추가
        city_distances = [(j, dist) for j, dist in enumerate(distances)]
        # 현재 도시를 제외하고 정렬하여 가장 가까운 도시부터 순서대로 저장
        city_distances.sort(key=lambda x: x[1])
        # 가장 가까운 num_neighbors 개의 도시를 선택하여 근접 도시 목록에 추가
        proximity_list.append([city[0] for city in city_distances[1:num_neighbors+1]])

    return proximity_list

# 근접 도시 목록 생성
proximity_list = create_proximity_list(distance_matrix)


def assign_to_quadrant(x, y, city, visit):
    unvisit = set(range(998)) - visit
    q1 = []
    q2 = []
    q3 = []
    q4 = []
    h_point = [] #후에 휴리스틱값에 기본이 될 좌표 
    
    # 현재도시에서 x축, y축 나눈 곳에서 제일 먼 점 찾기
    for n in unvisit:
        x1, y1 = city[n]
        if x1 > x and y1 > y:
            q1.append((x1, y1))
        elif x1 < x and y1 > y:
            q2.append((x1, y1))
        elif x1< x and y1 < y:
            q3.append((x1, y1))
        elif x1> x and y1 < y:
            q4.append((x1, y1))

    # 각 사분면에서 가장 먼 도시를 찾음
    for quadrant in [q1, q2, q3, q4]:
        max_length = 0
        farthest_city = None
        for city in quadrant:
            length = distance(city, (x, y))
            if length > max_length:
                max_length = length
                farthest_city = city
        h_point.append(farthest_city)

    return h_point

def heuristic(x, y):
    farthest_city =assign_to_quadrant(x, y)
    count = len(farthest_city)
    if count == 1:
        return distance((x, y), farthest_city[0])
    elif count == 2:
        # 두 가지 경로 중 더 짧은 경로를 선택
        path_1 = distance((x, y), farthest_city[0]) + distance(farthest_city[0], farthest_city[1])
        path_2 = distance((x, y), farthest_city[1]) + distance(farthest_city[1], farthest_city[0])
        return min(path_1, path_2)
    elif count == 3:
        # 각 도시를 방문하고 나머지 두 도시 간의 거리를 비교
        path_1 = distance((x, y), farthest_city[0]) +\
            min(distance(farthest_city[1], farthest_city[0]), distance(farthest_city[2], farthest_city[0]))+\
                distance(farthest_city[1], farthest_city[2])
        path_2 = distance((x, y), farthest_city[1]) + \
                 min(distance(farthest_city[0], farthest_city[1]), distance(farthest_city[2], farthest_city[1])) + \
                 distance(farthest_city[0], farthest_city[2])
        path_3 = distance((x, y), farthest_city[2]) + \
                 min(distance(farthest_city[0], farthest_city[2]), distance(farthest_city[1], farthest_city[2])) + \
                 distance(farthest_city[0], farthest_city[1])
        
        # 최소 거리를 반환
        return min(path_1, path_2, path_3)
    elif count == 4:
        # 각 시작점에서 다른 세 도시까지의 최소 거리와 두 도시간의 연결 거리를 계산
        paths = []
        for i in range(4):
            current_to_city = distance((x, y), farthest_city[i])
            other_cities = [farthest_city[j] for j in range(4) if j != i]
            min_distance_to_others = min(
                distance(farthest_city[i], other_cities[0]),
                distance(farthest_city[i], other_cities[1]),
                distance(farthest_city[i], other_cities[2])
            )
            # 남은 도시들 간의 거리
            remaining_distances = [
                distance(other_cities[0], other_cities[1]),
                distance(other_cities[1], other_cities[2]),
                distance(other_cities[2], other_cities[0])
            ]
            total_path_distance = current_to_city + min_distance_to_others + min(remaining_distances)
            paths.append(total_path_distance)

        # 모든 경로 중 최소값 반환
        return min(paths)

def child(proximity_list, dist_matrix, city_coordinates, mutate=0.03):
    p_city = 0  # 현재 도시
    r_city = set(range(1, len(city_coordinates)))  # 남은 도시

    total_distance = 0
    path = [0]  # 시작 도시를 포함한 경로
    while len(r_city) > 0:
        min_distance = float('inf')
        next_city = None

        # 현재 도시와 근접한 도시 간의 거리에 반비례하는 확률을 계산하여 다음 도시 선택
        probabilities = []
        for city in proximity_list[p_city]:
            if city in r_city:  # 방문하지 않은 도시인 경우에만 고려
                # 다음 도시까지의 거리
                dist_to_city = dist_matrix[p_city][city]
                # 현재 도시에서 다음 도시까지의 휴리스틱 값
                heuristic_value = heuristic(city_coordinates[city][0], city_coordinates[city][1])
                # 현재까지의 거리 + 휴리스틱 값을 고려한 거리
                total_cost = dist_to_city + heuristic_value

                # 현재 도시와 다음 도시 간의 거리에 반비례하는 확률을 계산하여 리스트에 추가
                probabilities.append(1 / total_cost)

        # 확률을 기반으로 다음 도시 선택
        next_city = rd.choices(list(r_city), probabilities)[0]

        # 다음 도시로 이동
        total_distance += dist_matrix[p_city][next_city]
        p_city = next_city
        r_city.remove(next_city)
        path.append(next_city)

        # 일정 확률로 돌연변이 발생
        if rd.random() < mutate:
            mutated_path, _ = mutation(path, dist_matrix)
            # 돌연변이를 적용한 후 경로 업데이트
            path = mutated_path

    # 시작 도시로 다시 돌아가는 경로 추가
    path.append(0)
    total_distance += dist_matrix[p_city][0]

    return total_distance, path

def genetic_algo(city_list, distance_matrix, proximity_list, population_size=5, generations=30, mutation_rate=0.05):
    # 초기 인구 생성: 각 경로는 0부터 시작하여 도시 수만큼의 인덱스를 무작위로 섞은 리스트입니다.
    population = [np.random.permutation(len(city_list)).tolist() for _ in range(population_size)]
    for individual in population:
        individual.insert(0, individual.pop(individual.index(0)))  # 모든 경로가 0에서 시작하도록 조정

    best_solution = None
    best_cost = float('inf')

    for generation in range(generations):
        # 새 세대의 인구
        new_population = []
        costs = []

        # 기존 인구에 대해 새로운 자식 생성
        for _ in range(population_size):
            cost_val, child_path = child(proximity_list, distance_matrix, city_list)
            if rd.random() < mutation_rate:
                child_path, _ = mutation(child_path, distance_matrix)
            child_path.insert(0, child_path.pop(child_path.index(0)))  # 경로 시작 조정
            new_population.append(child_path)
            costs.append(cost_val)

        # 기존 인구와 새로운 자식 인구를 합쳐 최상위 5개만 선택
        combined = list(zip(new_population, costs))
        combined.sort(key=lambda x: x[1])  # 비용에 따라 정렬
        population = [x[0] for x in combined[:population_size]]  # 상위 5개 선택
        current_best_cost = combined[0][1]

        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_solution = combined[0][0]

        print(f"Generation {generation}: Best Cost = {best_cost}")

    return best_solution, best_cost

def write_to_csv(file_name, path):
    """주어진 경로를 CSV 파일로 저장합니다."""
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[city] for city in path])

def main():
    csvfile = '2024_AI_TSP.csv'
    city_list = read_csv(csvfile)  # CSV 파일에서 도시 목록을 읽어옴
    distance_matrix = create_distance_matrix(city_list)  # 거리 행렬 생성
    proximity_list = create_proximity_list(distance_matrix)  # 근접도 목록 생성

    # 유전 알고리즘 실행
    best_path, best_path_cost = genetic_algo(city_list, distance_matrix, proximity_list)

    # 결과 출력
    print("최적 경로의 비용:", best_path_cost)
    print("최적 경로:", best_path)

    # 결과를 CSV 파일로 저장
    write_to_csv('ga_solution.csv', best_path)

if __name__ == "__main__":
    main()