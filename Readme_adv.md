def mutate(genotype: Genotype, μ: float) -> Genotype:
    g' = genotype.clone()
    
    # Structural mutations
    if rand() < μ:
        g'.modules.append(generate_random_module())  # addition
    if rand() < μ and len(g'.modules) > 3:
        g'.modules.remove(random.choice(g'.modules))  # deletion
    
    # Parametric mutations (continuous)
    for m in g'.modules:
        m.hyperparams += N(0, μ * σ_hyper)
    
    # Topological mutations  
    if rand() < μ:
        add_random_connection(g')
    if rand() < μ:
        remove_connection(g', method='low_importance')
    
    # Plasticity mutations
    for p in g'.plasticity_rules:
        p.metaparams['η'] *= exp(N(0, μ))
        if rand() < μ:
            p.rule_type = mutate_rule_type(p.rule_type)
    
    return g'
