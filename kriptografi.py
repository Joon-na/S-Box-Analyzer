import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
from io import BytesIO

# Utility functions
def convert_sbox_to_binary(sbox, bit_length):
    return np.array([[int(bit) for bit in format(value, f'0{bit_length}b')] for value in sbox])

def apply_walsh_hadamard_transformation(func):
    size = len(func)
    transformed = np.copy(func) * 2 - 1
    step = 1
    while step < size:
        for i in range(0, size, step * 2):
            for j in range(step):
                u = transformed[i + j]
                v = transformed[i + j + step]
                transformed[i + j] = u + v
                transformed[i + j + step] = u - v
        step *= 2
    return transformed

# Analysis functions
def calculate_nonlinearity(binary_sbox, num_bits, num_outputs):
    nl_values = []
    for i in range(num_outputs):
        func = binary_sbox[:, i]
        wht = apply_walsh_hadamard_transformation(func)
        max_correlation = np.max(np.abs(wht))
        nl = 2**(num_bits-1) - (max_correlation // 2)
        nl_values.append(nl)
    return min(nl_values), nl_values

def calculate_sac(binary_sbox, num_bits, num_outputs):
    num_inputs = 2**num_bits
    sac_matrix = np.zeros((num_bits, num_outputs))
    for i in range(num_inputs):
        original_output = binary_sbox[i]
        for bit in range(num_bits):
            flipped_input = i ^ (1 << bit)
            if flipped_input < num_inputs:  # Ensure we're not accessing out of bounds
                flipped_output = binary_sbox[flipped_input]
                bit_changes = original_output ^ flipped_output
                sac_matrix[bit] += bit_changes
    sac_matrix /= num_inputs
    return np.mean(sac_matrix), sac_matrix

def calculate_bic_nl(binary_sbox, num_bits, num_outputs):
    bic_nl_values = []
    for i in range(num_outputs):
        for j in range(i + 1, num_outputs):
            combined_func = binary_sbox[:, i] ^ binary_sbox[:, j]
            wht = apply_walsh_hadamard_transformation(combined_func)
            max_correlation = np.max(np.abs(wht))
            nl = 2**(num_bits-1) - (max_correlation // 2)
            bic_nl_values.append(nl)
    return min(bic_nl_values), bic_nl_values

def calculate_bic_sac(binary_sbox, num_bits, num_outputs):
    bic_sac_values = []
    num_inputs = 2**num_bits
    for i in range(num_outputs):
        for j in range(i + 1, num_outputs):
            sac_matrix = np.zeros(num_bits)
            for k in range(num_bits):
                count = 0
                for x in range(num_inputs):
                    flipped_x = x ^ (1 << k)
                    if flipped_x < num_inputs:  # Ensure we're not accessing out of bounds
                        original = binary_sbox[x, i] ^ binary_sbox[x, j]
                        flipped = binary_sbox[flipped_x, i] ^ binary_sbox[flipped_x, j]
                        if original != flipped:
                            count += 1
                sac_matrix[k] = count / num_inputs
            bic_sac_values.append(np.mean(sac_matrix))
    return np.mean(bic_sac_values), bic_sac_values

def calculate_lap(sbox, num_bits):
    num_inputs = 2**num_bits
    bias_table = np.zeros((num_inputs, num_inputs))
    for a in range(1, num_inputs):
        for b in range(num_inputs):
            count = 0
            for x in range(num_inputs):
                input_dot = bin(x & a).count('1') % 2
                output_dot = bin(sbox[x] & b).count('1') % 2
                if input_dot == output_dot:
                    count += 1
            bias_table[a, b] = abs(count - num_inputs // 2)
    lap_max = np.max(bias_table) / num_inputs
    return lap_max, bias_table

def calculate_dap(sbox, num_bits):
    num_inputs = 2**num_bits
    diff_table = np.zeros((num_inputs, num_inputs), dtype=int)
    for delta_x in range(1, num_inputs):
        for x in range(num_inputs):
            x_prime = x ^ delta_x
            delta_y = sbox[x] ^ sbox[x_prime]
            diff_table[delta_x, delta_y] += 1
    dap_max = np.max(diff_table) / num_inputs
    return dap_max, diff_table

# Main Streamlit app
def main():
    st.set_page_config(page_title="S-Box Analysis Tool", page_icon="🔐", layout="wide")

    # Custom CSS for improved dark mode design
    st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .main-title {
        color: #bb86fc;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .section-title {
        color: #03dac6;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #bb86fc;
        color: #1e1e1e;
        font-weight: 600;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #03dac6;
    }
    .stProgress>div>div>div {
        background-color: #03dac6;
    }
    .stDataFrame {
        border: 1px solid #333333;
        border-radius: 0.375rem;
    }
    .stExpander {
        border: 1px solid #333333;
        border-radius: 0.375rem;
        background-color: #2d2d2d;
    }
    .streamlit-expanderHeader {
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    .streamlit-expanderContent {
        background-color: #2d2d2d;
    }
    .stCheckbox>label {
        color: #e0e0e0;
    }
    .stFileUploader>label {
        color: #e0e0e0;
    }
    .stFileUploader>div>div {
        background-color: #2d2d2d;
        border: 1px solid #333333;
    }
    .css-1cpxqw2 {
        color: #e0e0e0;
        background-color: #2d2d2d;
    }
    .css-1cpxqw2:hover {
        background-color: #3d3d3d;
    }
    .css-1cpxqw2:focus:not(:active) {
        border-color: #bb86fc;
        color: #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">🔐 S-Box Analysis Tool</h1>', unsafe_allow_html=True)

    # File upload section
    st.markdown('<h2 class="section-title">Upload S-Box</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

    # Provide sample S-Box for download
    with st.expander("Download Sample S-Box"):
        example_sbox = pd.DataFrame(
            [99, 77, 116, 121, 232, 43, 110, 44, 199, 231, 38, 172, 229, 223, 165, 164,
            205, 177, 143, 220, 50, 237, 176, 66, 101, 53, 56, 74, 195, 42, 2, 69,
            85, 201, 10, 131, 21, 119, 221, 151, 203, 1, 36, 181, 138, 115, 146, 41,
            71, 78, 14, 210, 84, 183, 134, 32, 52, 124, 108, 118, 18, 238, 204, 230,
            25, 5, 54, 83, 215, 7, 19, 137, 57, 0, 8, 39, 93, 139, 120, 104,
            127, 48, 169, 135, 242, 12, 6, 31, 4, 28, 126, 227, 207, 243, 241, 47,
            113, 29, 148, 250, 180, 125, 187, 35, 153, 142, 9, 130, 240, 23, 163, 144,
            219, 30, 68, 149, 198, 55, 59, 147, 197, 170, 189, 89, 95, 98, 128, 251,
            63, 87, 49, 253, 168, 252, 26, 236, 88, 158, 81, 245, 58, 100, 22, 20,
            244, 96, 75, 72, 167, 206, 129, 247, 76, 51, 234, 166, 255, 178, 90, 17,
            109, 193, 171, 182, 103, 235, 112, 117, 202, 226, 212, 16, 209, 37, 60, 150,
            159, 80, 157, 33, 122, 160, 73, 132, 174, 65, 224, 61, 217, 97, 185, 225,
            11, 156, 92, 190, 152, 140, 175, 79, 233, 123, 13, 106, 15, 191, 67, 254,
            228, 200, 114, 141, 162, 133, 45, 136, 62, 186, 3, 196, 111, 213, 34, 161,
            94, 216, 188, 249, 145, 179, 24, 154, 208, 239, 40, 211, 46, 222, 27, 102,
            214, 86, 194, 82, 184, 192, 218, 105, 91, 246, 64, 107, 173, 155, 248, 70])
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            example_sbox.to_excel(writer, index=False, sheet_name='S-Box', header=False)
        
        st.download_button(
            label="Download Sample S-Box",
            data=excel_buffer,
            file_name="sbox_sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Main content
    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=None)
        st.markdown('<h2 class="section-title">Uploaded S-Box</h2>', unsafe_allow_html=True)
        st.dataframe(df)

        sbox = df.values.flatten()
        num_bits = int(np.log2(len(sbox)))
        num_outputs = 8  # Assuming 8-bit output
        binary_sbox = convert_sbox_to_binary(sbox, num_outputs)

        # Test selection
        st.markdown('<h2 class="section-title">Select Analysis Tests</h2>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            nl_test = st.checkbox("Nonlinearity (NL)", value=True)
            sac_test = st.checkbox("SAC", value=True)
        with col2:
            bic_nl_test = st.checkbox("BIC-NL", value=True)
            bic_sac_test = st.checkbox("BIC-SAC", value=True)
        with col3:
            lap_test = st.checkbox("LAP", value=True)
            dap_test = st.checkbox("DAP", value=True)

        if st.button("Run Analysis"):
            results = {}
            progress_bar = st.progress(0)
            status_text = st.empty()

            tests = [
                (nl_test, "Nonlinearity (NL)", lambda: calculate_nonlinearity(binary_sbox, num_bits, num_outputs)),
                (sac_test, "SAC", lambda: calculate_sac(binary_sbox, num_bits, num_outputs)),
                (bic_nl_test, "BIC-NL", lambda: calculate_bic_nl(binary_sbox, num_bits, num_outputs)),
                (bic_sac_test, "BIC-SAC", lambda: calculate_bic_sac(binary_sbox, num_bits, num_outputs)),
                (lap_test, "LAP", lambda: calculate_lap(sbox, num_bits)),
                (dap_test, "DAP", lambda: calculate_dap(sbox, num_bits))
            ]

            total_tests = sum(test[0] for test in tests)
            completed_tests = 0

            for selected, name, func in tests:
                if selected:
                    status_text.text(f"Running {name} analysis...")
                    results[name] = func()
                    completed_tests += 1
                    progress_bar.progress(completed_tests / total_tests)

            status_text.text("Analysis complete!")

            # Display results
            st.markdown('<h2 class="section-title">Analysis Results</h2>', unsafe_allow_html=True)
            for test, result in results.items():
                with st.expander(f"{test} Results"):
                    if isinstance(result, tuple):
                        st.write(f"Main value: {result[0]}")
                        st.write("Detailed results:")
                        if isinstance(result[1], np.ndarray):
                            st.dataframe(pd.DataFrame(result[1]))
                        else:
                            st.write(result[1])
                    else:
                        st.write(result)

            # Prepare results for download
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                for test, result in results.items():
                    if isinstance(result, tuple):
                        pd.DataFrame({"Main Value": [result[0]]}).to_excel(writer, sheet_name=f"{test}_Main", index=False)
                        if isinstance(result[1], np.ndarray):
                            pd.DataFrame(result[1]).to_excel(writer, sheet_name=f"{test}_Detailed", index=False)
                        elif isinstance(result[1], list):
                            pd.DataFrame({"Values": result[1]}).to_excel(writer, sheet_name=f"{test}_Detailed", index=False)
                    else:
                        pd.DataFrame({test: [result]}).to_excel(writer, sheet_name=test, index=False)

            st.download_button(
                label="Download Results",
                data=output.getvalue(),
                file_name="sbox_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("Please upload an S-Box file to begin analysis.")

if __name__ == "__main__":
    main()