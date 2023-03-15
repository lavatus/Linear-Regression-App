def main():
    from Linear_Regression import Linear_Regression
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    #st.set_page_config(layout="wide")

    st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    title = '<p class="big-font">Example of Linear Regression !!!</p>'
    st.markdown(title, unsafe_allow_html=True)

    values_X  = st.slider(
        'Select a range of X values',
        -20, 20, (-5, 5))

    w =  st.number_input('Select w value',  min_value=1.0, max_value=20., value=5.0, step=1.0)
    b =  st.number_input('Select b value',  min_value=1.0, max_value=20., value=3.0, step=1.0)


    lower_bound = values_X[0]
    upper_bound = values_X[1]

    X = np.arange(lower_bound, upper_bound, step = 0.5)
    noise = np.random.rand(len(X)) * (upper_bound - lower_bound)
    Y = w*X + b + noise


    N_epochs = st.number_input("Number of epochs", min_value=1, max_value=100, value=20, step=1)
    lr = st.number_input("Learning rate", min_value=0.001, max_value=10.0, value=0.01, step=0.001)

    state = st.button('Generate predicted line')
    fig, ax  = plt.subplots()
    the_plot = st.pyplot(plt)
    the_plot.pyplot(plt)
    if state:
        model = Linear_Regression(X, Y)
        for epoch in range(N_epochs):
            model.update_parameters(lr)       
            Y_pred = model.predict()
            fig, ax  = plt.subplots()
            ax.plot(X,Y, 'ro')
            ax.plot(X,Y_pred)
            time.sleep(0.01)
            the_plot.pyplot(plt)
        w_pred, b_pred = model.get_weights()
        st.write(f'Predicted line is {w_pred :0.2f} * x + {b_pred:0.2f}')
if __name__ == '__main__':
    main()