# df_final = df_final.sort_values(["bloque_tiempo", "ms"]).reset_index(drop=True)

# folds = []
# for i, (idx_ent, idx_prueba) in enumerate(tscv.split(df_final), start=1):
#     t = df_final["bloque_tiempo"]
#     folds.append({
#         "fold": i,
#         "entrenamiento": f"{t.iloc[idx_ent[0]].date()} a {t.iloc[idx_ent[-1]].date()}",
#         "prueba": f"{t.iloc[idx_prueba[0]].date()} a {t.iloc[idx_prueba[-1]].date()}",
#         "bloques_entrenamiento": len(idx_ent),
#         "bloques_prueba": len(idx_prueba),
#     })
# display(pd.DataFrame(folds))
