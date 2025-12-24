# 🛒 AI-Powered E-Commerce Recommendation System

A modern, machine-learning–based product recommendation system built using **Python** and **Streamlit**.  
This project demonstrates how personalized and popularity-based recommendations can be integrated into an e-commerce platform.

---

## 🚀 Features

- Cold-start handling for **new visitors**
- Personalized recommendations for **returning users**
- Category-based product discovery
- Modern card-based UI (e-commerce style)
- Collaborative Filtering using **Matrix Factorization (FunkSVD)**
- Single-file full-stack implementation (`app.py`)

---

## 🧠 How It Works

### New Visitor
- No prior interaction data
- Recommendations are based on **popular and trending products**

### Returning Customer
- Uses **collaborative filtering**
- Recommendations are generated using a trained **FunkSVD model**
- Products are selected based on similar user behavior

A product catalog layer is used to enrich recommendations with:
- Product name
- Brand
- Category
- Price
- Rating

---

## 🏗️ Project Structure

