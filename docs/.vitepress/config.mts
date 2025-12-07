import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "num.zig",
  description: "A fast, high-performance numerical computing library for Zig.",
  base: '/num.zig/',
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/quick-start' },
      { text: 'API', link: '/api/overview' }
    ],

    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'Quick Start', link: '/guide/quick-start' },
          { text: 'Data Types', link: '/guide/datatypes' },
          { text: 'NDArray', link: '/guide/ndarray' },
          { text: 'Broadcasting', link: '/guide/broadcasting' },
          { text: 'Operations', link: '/guide/operations' },
          { text: 'Linear Algebra', link: '/guide/linalg' },
          { text: 'Statistics', link: '/guide/stats' },
          { text: 'Random', link: '/guide/random' },
          { text: 'FFT', link: '/guide/fft' },
          { text: 'Machine Learning', link: '/guide/ml' },
          { text: 'Polynomials', link: '/guide/poly' },
          { text: 'Signal Processing', link: '/guide/signal' },
          { text: 'Set Operations', link: '/guide/setops' },
          { text: 'Algorithms', link: '/algorithms' },
          { text: 'Input / Output', link: '/guide/io' }
        ]
      },
      {
        text: 'API Reference',
        items: [
          { text: 'Overview', link: '/api/overview' },
          { text: 'Core', link: '/api/core' },
          { text: 'Math', link: '/api/math' },
          { text: 'Indexing', link: '/api/indexing' },
          { text: 'Manipulation', link: '/api/manipulation' },
          { text: 'Linear Algebra', link: '/api/linalg' },
          { text: 'Statistics', link: '/api/stats' },
          { text: 'Random', link: '/api/random' },
          { text: 'Polynomials', link: '/api/poly' },
          { text: 'Signal Processing', link: '/api/signal' },
          { text: 'FFT', link: '/api/fft' },
          { text: 'Complex Numbers', link: '/api/complex' },
          { text: 'Set Operations', link: '/api/setops' },
          { text: 'Sorting', link: '/api/sort' },
          { text: 'Interpolation', link: '/api/interpolate' },
          { text: 'Sparse Matrices', link: '/api/sparse' },
          { text: 'Autograd', link: '/api/autograd' },
          { text: 'DataFrame', link: '/api/dataframe' },
          { text: 'Optimization', link: '/api/optimize' },
          { text: 'Machine Learning', link: '/api/ml' },
          { text: 'Input / Output', link: '/api/io' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/muhammad-fiaz/num.zig' }
    ],

    footer: {
      message: 'Released under the Apache 2.0 License.',
      copyright: 'Copyright © 2025 Muhammad Fiaz'
    }
  }
})
