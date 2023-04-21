from transppi.ppi_transformer import PPITransformer

if __name__ == '__main__':
    model = PPITransformer(dim_hidden=32, dim_edge_feat=64, dim_vertex_feat=1024)
    cnt = 0
    for parameter in model.parameters():
        prod = 1
        for i in parameter.shape:
            prod *= i
        cnt += prod
        print(parameter.shape)
    print(cnt)