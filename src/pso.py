import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
import random
import warnings
warnings.filterwarnings('ignore')

class PSOFeatureSelection:
    """
    Implementación de Particle Swarm Optimization (PSO) para selección de características
    basada en el paper de personalidad con SMOTETomek
    """
    def __init__(self, n_particles=20, max_iterations=50, w=0.5, c1=2.0, c2=2.0, 
                 min_features=5, random_state=42):
        """
        Args:
            n_particles: Número de partículas en el enjambre
            max_iterations: Número máximo de iteraciones
            w: Peso de inercia
            c1, c2: Constantes de aceleración (learning factors)
            min_features: Número mínimo de características a seleccionar
            random_state: Semilla para reproducibilidad
        """
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.min_features = min_features
        self.random_state = random_state
        self.best_features_ = None
        self.best_fitness_ = -np.inf
        self.fitness_history_ = []
        
    def _initialize_particles(self, n_features):
        """Inicializa las partículas con posiciones y velocidades aleatorias"""
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        # Posiciones: 0 o 1 para cada característica (binario)
        positions = np.random.randint(0, 2, (self.n_particles, n_features))
        
        # Asegurar que cada partícula tenga al menos min_features características
        for i in range(self.n_particles):
            if np.sum(positions[i]) < self.min_features:
                # Seleccionar aleatoriamente min_features características
                indices = np.random.choice(n_features, self.min_features, replace=False)
                positions[i] = 0
                positions[i][indices] = 1
        
        # Velocidades: valores continuos entre -4 y 4
        velocities = np.random.uniform(-4, 4, (self.n_particles, n_features))
        
        return positions, velocities
    
    def _evaluate_fitness(self, X, y, features_mask, classifier):
        """Evalúa la fitness de una partícula usando F1-score con validación cruzada"""
        if np.sum(features_mask) == 0:
            return -1  # Penalizar si no hay características seleccionadas
        
        try:
            X_selected = X[:, features_mask.astype(bool)]
            
            # Usar F1-score como métrica de fitness (como en el paper)
            scorer = make_scorer(f1_score, average='weighted')
            scores = cross_val_score(classifier, X_selected, y, cv=5, scoring=scorer, n_jobs=-1)
            fitness = np.mean(scores)
            
            # Penalizar si hay muy pocas características
            if np.sum(features_mask) < self.min_features:
                fitness *= 0.5
                
            return fitness
        except Exception as e:
            return -1  # Penalizar en caso de error
    
    def _sigmoid(self, x):
        """Función sigmoide para convertir velocidades a probabilidades"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def fit(self, X, y, classifier=None):
        """
        Ejecuta el algoritmo PSO para encontrar las mejores características
        
        Args:
            X: Matriz de características
            y: Vector objetivo
            classifier: Clasificador a usar para evaluación (por defecto RandomForest)
        """
        if classifier is None:
            classifier = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
        
        n_features = X.shape[1]
        
        # Inicializar partículas
        positions, velocities = self._initialize_particles(n_features)
        
        # Mejores posiciones individuales y globales
        pbest_positions = positions.copy()
        pbest_fitness = np.full(self.n_particles, -np.inf)
        gbest_position = None
        gbest_fitness = -np.inf
        
        print(f"Iniciando PSO con {self.n_particles} partículas, {self.max_iterations} iteraciones")
        print(f"Evaluando {n_features} características...")
        
        # Evaluar fitness inicial
        for i in range(self.n_particles):
            fitness = self._evaluate_fitness(X, y, positions[i], classifier)
            pbest_fitness[i] = fitness
            
            if fitness > gbest_fitness:
                gbest_fitness = fitness
                gbest_position = positions[i].copy()
        
        self.fitness_history_.append(gbest_fitness)
        
        # Iteraciones principales del PSO
        for iteration in range(self.max_iterations):
            for i in range(self.n_particles):
                # Generar números aleatorios r1 y r2
                r1 = np.random.random(n_features)
                r2 = np.random.random(n_features)
                
                # Actualizar velocidad según ecuación (1) del paper
                velocities[i] = (self.w * velocities[i] + 
                               self.c1 * r1 * (pbest_positions[i] - positions[i]) +
                               self.c2 * r2 * (gbest_position - positions[i]))
                
                # Aplicar límites de velocidad
                velocities[i] = np.clip(velocities[i], -4, 4)
                
                # Actualizar posición usando sigmoide para conversión binaria
                probabilities = self._sigmoid(velocities[i])
                positions[i] = (np.random.random(n_features) < probabilities).astype(int)
                
                # Asegurar número mínimo de características
                if np.sum(positions[i]) < self.min_features:
                    # Mantener las características actuales y agregar más
                    remaining = self.min_features - np.sum(positions[i])
                    available_indices = np.where(positions[i] == 0)[0]
                    if len(available_indices) >= remaining:
                        selected_indices = np.random.choice(available_indices, remaining, replace=False)
                        positions[i][selected_indices] = 1
                
                # Evaluar nueva posición
                fitness = self._evaluate_fitness(X, y, positions[i], classifier)
                
                # Actualizar pbest
                if fitness > pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest_positions[i] = positions[i].copy()
                
                # Actualizar gbest
                if fitness > gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = positions[i].copy()
            
            self.fitness_history_.append(gbest_fitness)
            
            if (iteration + 1) % 10 == 0:
                n_selected = np.sum(gbest_position)
                print(f"Iteración {iteration + 1}: Mejor fitness = {gbest_fitness:.4f}, "
                      f"Características seleccionadas = {n_selected}")
        
        # Guardar mejores resultados
        self.best_features_ = gbest_position.astype(bool)
        self.best_fitness_ = gbest_fitness
        
        n_selected = np.sum(self.best_features_)
        print(f"\nPSO completado:")
        print(f"Mejor fitness: {self.best_fitness_:.4f}")
        print(f"Características seleccionadas: {n_selected}/{n_features}")
        
        return self
    
    def transform(self, X):
        """Aplica la selección de características a los datos"""
        if self.best_features_ is None:
            raise ValueError("PSO no ha sido entrenado. Ejecute fit() primero.")
        
        return X[:, self.best_features_]
    
    def fit_transform(self, X, y, classifier=None):
        """Entrena PSO y aplica la transformación"""
        return self.fit(X, y, classifier).transform(X)


def prepare_data_with_pso_smotetomek(df: pd.DataFrame, algoritmo: str, 
                                   apply_pca=False, n_components=None, 
                                   modo="multiclase", use_pso=True, 
                                   pso_params=None):
    """
    Prepara los datos para el entrenamiento implementando PSO + SMOTETomek
    según la metodología del paper de reconocimiento de personalidad.
    
    Args:
        df: DataFrame con datos preprocesados
        algoritmo: Nombre del algoritmo para logging
        apply_pca: Si True, aplica PCA para reducción de dimensionalidad
        n_components: Componentes PCA o varianza explicada si es float
        modo: 'multiclase' o 'binario'
        use_pso: Si True, aplica PSO para selección de características
        pso_params: Parámetros del PSO (dict)
    
    Returns:
        X, y: Conjuntos de datos preparados para entrenamiento
    """
    print("=== Preparando datos con PSO + SMOTETomek ===")
    
    # 1. Validación inicial
    required_cols = ['apnea', 'apnea_severity_ordinal']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Variables faltantes: {missing_cols}")
    
    # 2. Definir variable objetivo
    if modo == 'multiclase':
        y = df['apnea_severity_ordinal'].copy()
        print(f"Modo multiclase - Distribución original:")
        print(y.value_counts().sort_index())
    elif modo == 'binario':
        y = df['apnea'].copy()
        print(f"Modo binario - Distribución original:")
        print(y.value_counts().sort_index())
    else:
        raise ValueError("Modo debe ser 'multiclase' o 'binario'")
    
    # 3. Preparar variables predictoras
    variables_a_excluir = [
        'apnea', 'nsrr_ahi_hp4u_aasm15', 'apnea_severity', 'apnea_severity_ordinal',
        'nsrr_ahi_hp3r_aasm15', 'apnea_significativa', 'apnea_severa'
    ]
    
    variables_categoricas = ['bmi_categoria', 'categoria_pa']
    
    # Buscar variables categóricas adicionales
    for col in df.columns:
        if (df[col].dtype == 'object' or df[col].dtype == 'category') and col not in variables_a_excluir:
            if col not in variables_categoricas:
                variables_categoricas.append(col)
    
    X = df.copy()
    
    # Eliminar variables objetivo
    for var in variables_a_excluir:
        if var in X.columns:
            X = X.drop(var, axis=1)
    
    # 4. Procesamiento de variables categóricas (como en el código original)
    print("Procesando variables categóricas...")
    for var in variables_categoricas:
        if var in X.columns:
            unique_cats = X[var].nunique()
            if unique_cats > 10:
                print(f"Advertencia: {var} tiene {unique_cats} categorías únicas.")
                top_cats = X[var].value_counts().head(8).index
                X[var] = X[var].apply(lambda x: x if x in top_cats else 'Otros')
            
            try:
                dummies = pd.get_dummies(X[var], prefix=var, drop_first=True, dtype=int)
                X = pd.concat([X, dummies], axis=1)
                X = X.drop(var, axis=1)
            except Exception as e:
                print(f"Error procesando {var}: {e}")
                X = X.drop(var, axis=1)
    
    # Eliminar columnas no numéricas
    non_numeric_cols = []
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        print(f"Eliminando columnas no numéricas: {non_numeric_cols}")
        X = X.drop(columns=non_numeric_cols)
    
    # Convertir a numérico
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            X = X.drop(columns=[col])
    
    # 5. Transformación de datos sesgados
    print("Aplicando transformaciones para mejorar distribuciones...")
    skewed_features = []
    for col in X.select_dtypes(include=[np.number]).columns:
        if abs(X[col].skew()) > 2:
            skewed_features.append(col)
    
    if skewed_features:
        print(f"Aplicando transformación Yeo-Johnson a {len(skewed_features)} variables")
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        X[skewed_features] = pt.fit_transform(X[skewed_features])
    
    # 6. Aplicar PCA si se solicita (antes de PSO)
    if apply_pca:
        print(f"Aplicando PCA con componentes={n_components}...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if n_components is None:
            n_components = 0.95
        
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        if isinstance(n_components, float):
            n_cols = pca.n_components_
        else:
            n_cols = n_components
        
        X = pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(n_cols)],
            index=X.index
        )
        
        print(f"Varianza explicada total: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    # 7. NUEVA IMPLEMENTACIÓN: Balanceo con SMOTETomek (Paso 1 del pipeline)
    print("\n=== Paso 1: Balanceo de datos con SMOTETomek ===")
    
    # Verificar desbalance
    class_counts = y.value_counts().sort_index()
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"Ratio de desbalance original: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 1.5:
        try:
            # Aplicar SMOTETomek como en el paper
            smotetomek = SMOTETomek(
                smote=SMOTE(random_state=42, k_neighbors=5),
                tomek=TomekLinks(),
                random_state=42
            )
            
            X_balanced, y_balanced = smotetomek.fit_resample(X, y)
            
            # Convertir de vuelta a DataFrame/Series
            X = pd.DataFrame(X_balanced, columns=X.columns)
            y = pd.Series(y_balanced)
            
            print("Distribución después de SMOTETomek:")
            print(y.value_counts().sort_index())
            
        except Exception as e:
            print(f"SMOTETomek falló: {e}. Usando SMOTE básico.")
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            X = pd.DataFrame(X_balanced, columns=X.columns)
            y = pd.Series(y_balanced)
    
    # 8. NUEVA IMPLEMENTACIÓN: Selección de características con PSO (Paso 2 del pipeline)
    if use_pso and X.shape[1] > 10:  # Solo si hay suficientes características
        print(f"\n=== Paso 2: Optimización de características con PSO ===")
        
        # Configurar parámetros del PSO
        if pso_params is None:
            pso_params = {
                'n_particles': min(20, X.shape[1] // 2),
                'max_iterations': 30,
                'w': 0.5,
                'c1': 2.0,
                'c2': 2.0,
                'min_features': max(5, X.shape[1] // 10)
            }
        
        # Inicializar y ejecutar PSO
        pso = PSOFeatureSelection(**pso_params)
        
        # Estandarizar datos para PSO
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Ejecutar PSO
        pso.fit(X_scaled, y)
        
        # Aplicar selección de características
        selected_features = X.columns[pso.best_features_]
        X = X[selected_features]
        
        print(f"Características seleccionadas por PSO: {len(selected_features)}")
        print(f"Características seleccionadas: {list(selected_features[:10])}...")  # Mostrar primeras 10
    
    # 9. Limpieza final de datos
    X = X.select_dtypes(include=[np.number])
    
    if X.isnull().sum().sum() > 0:
        print("Rellenando valores NaN...")
        X = X.fillna(0)
    
    # Verificar valores infinitos
    try:
        if np.isinf(X.values).sum() > 0:
            print("Reemplazando valores infinitos...")
            X = X.replace([np.inf, -np.inf], 0)
    except TypeError:
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                if np.isinf(X[col]).sum() > 0:
                    X[col] = X[col].replace([np.inf, -np.inf], 0)
    
    # 10. Resumen final
    print(f"\n=== Resumen Final ===")
    print(f"Características finales: {X.shape[1]}")
    print(f"Muestras finales: {X.shape[0]}")
    print(f"Distribución final de clases:")
    print(y.value_counts().sort_index())
    
    return X, y


# # Función auxiliar para comparar métodos
# def compare_preparation_methods(df, algoritmo="SVM", modo="multiclase"):
#     """
#     Compara diferentes métodos de preparación de datos
#     """
#     print("=== COMPARACIÓN DE MÉTODOS DE PREPARACIÓN ===")
    
#     results = {}
    
#     # Método 1: Original (sin PSO)
#     print("\n1. Método Original (sin PSO)")
#     try:
#         X1, y1 = prepare_data(df, algoritmo, modo=modo)
#         results['Original'] = {'X_shape': X1.shape, 'y_dist': y1.value_counts().to_dict()}
#     except Exception as e:
#         print(f"Error en método original: {e}")
#         results['Original'] = {'error': str(e)}
    
#     # Método 2: Con SMOTETomek solamente
#     print("\n2. Método con SMOTETomek (sin PSO)")
#     try:
#         X2, y2 = prepare_data_with_pso_smotetomek(df, algoritmo, modo=modo, use_pso=False)
#         results['SMOTETomek'] = {'X_shape': X2.shape, 'y_dist': y2.value_counts().to_dict()}
#     except Exception as e:
#         print(f"Error con SMOTETomek: {e}")
#         results['SMOTETomek'] = {'error': str(e)}
    
#     # Método 3: Con PSO + SMOTETomek
#     print("\n3. Método con PSO + SMOTETomek")
#     try:
#         X3, y3 = prepare_data_with_pso_smotetomek(df, algoritmo, modo=modo, use_pso=True)
#         results['PSO+SMOTETomek'] = {'X_shape': X3.shape, 'y_dist': y3.value_counts().to_dict()}
#     except Exception as e:
#         print(f"Error con PSO+SMOTETomek: {e}")
#         results['PSO+SMOTETomek'] = {'error': str(e)}
    
#     return results


# # Ejemplo de uso
# """
# # Cargar datos preprocesados
# df = pd.read_csv('datos_preprocesados.csv')

# # Preparar datos con PSO + SMOTETomek
# X, y = prepare_data_with_pso_smotetomek(
#     df, 
#     algoritmo="SVM", 
#     modo="multiclase",
#     use_pso=True,
#     pso_params={
#         'n_particles': 20,
#         'max_iterations': 30,
#         'w': 0.5,
#         'c1': 2.0,
#         'c2': 2.0,
#         'min_features': 10
#     }
# )

# # Comparar métodos
# results = compare_preparation_methods(df, "SVM", "multiclase")
# print(results)
# """