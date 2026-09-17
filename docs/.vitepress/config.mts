import { defineConfig } from "vitepress";
import llmstxt from "vitepress-plugin-llms";

// Site configuration
export const SITE_URL = "https://muhammad-fiaz.github.io/num.zig";
export const SITE_NAME = "num.zig";
export const SITE_DESCRIPTION = "A fast, production-ready, high-performance numerical computing and N-dimensional array library for Zig with SIMD acceleration, linear algebra, FFT, statistics, multi-threaded CPU parallel execution, and portable NZIG v1.0 binary serialization.";

// Google Analytics and Google Tag Manager IDs
export const GA_ID = "G-6BVYCRK57P";
export const GTM_ID = "GTM-P4M9T8ZR";

// Google AdSense Client ID
export const ADSENSE_CLIENT_ID = "ca-pub-2040560600290490";

// SEO Keywords
export const KEYWORDS = "zig, numerical computing, ndarray, array, tensor, linear algebra, linalg, math, simd, parallel, cpu, statistics, fft, polynomial, sparse matrix, machine learning, nzig, binary serialization";

export default defineConfig({
  lang: "en-US",
  title: SITE_NAME,
  description: SITE_DESCRIPTION,
  base: "/num.zig/",
  lastUpdated: true,
  cleanUrls: false,

  sitemap: {
    hostname: `${SITE_URL}/`,
  },

  vite: {
    plugins: [llmstxt()],
  },

  head: [
    // Primary Meta Tags
    ["meta", { name: "title", content: SITE_NAME }],
    ["meta", { name: "description", content: SITE_DESCRIPTION }],
    ["meta", { name: "keywords", content: KEYWORDS }],
    ["meta", { name: "author", content: "Muhammad Fiaz" }],
    ["meta", { name: "robots", content: "index, follow" }],
    ["meta", { name: "language", content: "English" }],
    ["meta", { name: "revisit-after", content: "7 days" }],
    ["meta", { name: "generator", content: "VitePress" }],

    // Open Graph
    ["meta", { property: "og:type", content: "website" }],
    ["meta", { property: "og:url", content: SITE_URL }],
    ["meta", { property: "og:title", content: SITE_NAME }],
    ["meta", { property: "og:description", content: SITE_DESCRIPTION }],
    ["meta", { property: "og:image", content: `${SITE_URL}/cover.png` }],
    ["meta", { property: "og:image:width", content: "1536" }],
    ["meta", { property: "og:image:height", content: "1024" }],
    ["meta", { property: "og:image:alt", content: "num.zig - High Performance Numerical Computing for Zig" }],
    ["meta", { property: "og:image:secure_url", content: `${SITE_URL}/cover.png` }],
    ["meta", { property: "og:site_name", content: SITE_NAME }],
    ["meta", { property: "og:locale", content: "en_US" }],

    // Twitter Card
    ["meta", { name: "twitter:card", content: "summary_large_image" }],
    ["meta", { name: "twitter:url", content: SITE_URL }],
    ["meta", { name: "twitter:title", content: SITE_NAME }],
    ["meta", { name: "twitter:description", content: SITE_DESCRIPTION }],
    ["meta", { name: "twitter:image", content: `${SITE_URL}/cover.png` }],
    ["meta", { name: "twitter:image:alt", content: "num.zig - High Performance Numerical Computing for Zig" }],
    ["meta", { name: "twitter:site", content: "@muhammadfiaz_" }],
    ["meta", { name: "twitter:creator", content: "@muhammadfiaz_" }],

    // Canonical URL
    ["link", { rel: "canonical", href: SITE_URL }],

    // Favicons
    ["link", { rel: "icon", href: "/num.zig/favicon.ico" }],
    ["link", { rel: "icon", type: "image/png", sizes: "16x16", href: "/num.zig/favicon-16x16.png" }],
    ["link", { rel: "icon", type: "image/png", sizes: "32x32", href: "/num.zig/favicon-32x32.png" }],
    ["link", { rel: "apple-touch-icon", sizes: "180x180", href: "/num.zig/apple-touch-icon.png" }],
    ["link", { rel: "icon", type: "image/png", sizes: "192x192", href: "/num.zig/android-chrome-192x192.png" }],
    ["link", { rel: "icon", type: "image/png", sizes: "512x512", href: "/num.zig/android-chrome-512x512.png" }],
    ["link", { rel: "manifest", href: "/num.zig/site.webmanifest" }],

    // Theme color
    ["meta", { name: "theme-color", content: "#f7a41d" }],
    ["meta", { name: "msapplication-TileColor", content: "#f7a41d" }],

    // Google Analytics
    [
      "script",
      { async: "", src: `https://www.googletagmanager.com/gtag/js?id=${GA_ID}` },
    ],
    [
      "script",
      {},
      `window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', '${GA_ID}');`,
    ],

    // Google Tag Manager
    ...(GTM_ID
      ? ([
          [
            "script",
            {},
            `(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start': new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0], j=d.createElement(s), dl=l!='dataLayer'?'&l='+l:''; j.async=true; j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl; f.parentNode.insertBefore(j,f);})(window,document,'script','dataLayer','${GTM_ID}');`,
          ],
          [
            "noscript",
            {},
            `<iframe src="https://www.googletagmanager.com/ns.html?id=${GTM_ID}" height="0" width="0" style="display:none;visibility:hidden"></iframe>`,
          ],
        ] as [string, Record<string, string>, string][])
      : []),

    // Google AdSense
    [
      "script",
      {
        async: "",
        src: `https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${ADSENSE_CLIENT_ID}`,
        crossorigin: "anonymous",
      },
    ],
  ],

  transformPageData(pageData: any) {
    const pageTitle = pageData.title || SITE_NAME;
    const pageDescription = pageData.description || SITE_DESCRIPTION;
    const normalizedPath = pageData.relativePath
      .replace(/\.md$/, "")
      .replace(/(^|\/)index$/, "$1")
      .replace(/\/$/, "");
    const canonicalUrl = normalizedPath.length > 0 ? `${SITE_URL}/${normalizedPath}` : SITE_URL;

    pageData.frontmatter.head ??= [];
    pageData.frontmatter.head.push(
      ["link", { rel: "canonical", href: canonicalUrl }],
      ["meta", { property: "og:title", content: `${pageTitle} | ${SITE_NAME}` }],
      ["meta", { property: "og:url", content: canonicalUrl }]
    );

    if (pageData.frontmatter.description) {
      pageData.frontmatter.head.push(
        ["meta", { property: "og:description", content: pageData.frontmatter.description }],
        ["meta", { name: "description", content: pageData.frontmatter.description }]
      );
    }

    const isHome = pageData.relativePath === 'index.md';
    const lastUpdated = pageData.lastUpdated
      ? new Date(pageData.lastUpdated).toISOString()
      : new Date().toISOString();

    const graph: any[] = [];

    if (isHome) {
      graph.push({
        "@type": "WebSite",
        "name": SITE_NAME,
        "url": SITE_URL,
        "description": SITE_DESCRIPTION,
        "author": {
          "@type": "Person",
          "name": "Muhammad Fiaz",
          "url": "https://github.com/muhammad-fiaz"
        }
      });
    }

    const authorSchema = {
      "@type": "Person",
      "name": "Muhammad Fiaz",
      "url": "https://muhammadfiaz.com",
      "sameAs": [
        "https://github.com/muhammad-fiaz",
        "https://www.linkedin.com/in/muhammad-fiaz-",
        "https://x.com/muhammadfiaz_"
      ]
    };

    const primarySchema: Record<string, any> = {
      "@type": isHome ? "SoftwareApplication" : "TechArticle",
      "name": isHome ? SITE_NAME : pageTitle,
      "description": pageDescription,
      "url": canonicalUrl,
      "image": `${SITE_URL}/cover.png`,
      "author": authorSchema,
      "publisher": {
        "@type": "Organization",
        "name": "num.zig",
        "url": SITE_URL,
        "logo": {
          "@type": "ImageObject",
          "url": `${SITE_URL}/logo.png`
        }
      }
    };

    if (isHome) {
      Object.assign(primarySchema, {
        "applicationCategory": "DeveloperApplication",
        "operatingSystem": "Cross-platform",
        "programmingLanguage": "Zig",
        "offers": {
          "@type": "Offer",
          "price": "0",
          "priceCurrency": "USD"
        },
        "downloadUrl": "https://github.com/muhammad-fiaz/num.zig",
          "softwareVersion": "0.0.3",
        "license": "https://opensource.org/licenses/MIT"
      });
    } else {
      const pathParts = pageData.relativePath.split('/');
      const section = pathParts.length > 1
        ? pathParts[0].charAt(0).toUpperCase() + pathParts[0].slice(1)
        : 'Documentation';

      Object.assign(primarySchema, {
        "headline": pageTitle,
        "articleSection": section,
        "mainEntityOfPage": {
          "@type": "WebPage",
          "@id": canonicalUrl
        },
        "datePublished": "2026-01-01T00:00:00Z",
        "dateModified": lastUpdated
      });
    }
    graph.push(primarySchema);

    // BreadcrumbList Schema
    const breadcrumbs: any[] = [
      {
        "@type": "ListItem",
        "position": 1,
        "name": "Home",
        "item": SITE_URL
      }
    ];

    if (!isHome) {
      const pathParts = pageData.relativePath.replace(/\.md$/, '').split('/');
      let currentPath = SITE_URL;

      pathParts.forEach((part: string, index: number) => {
        currentPath += `/${part}`;
        const name = part.split('-').map(s => s.charAt(0).toUpperCase() + s.slice(1)).join(' ');

        breadcrumbs.push({
          "@type": "ListItem",
          "position": index + 2,
          "name": name,
          "item": index === pathParts.length - 1 ? canonicalUrl : currentPath
        });
      });
    }

    graph.push({
      "@type": "BreadcrumbList",
      "itemListElement": breadcrumbs
    });

    pageData.frontmatter.head.push([
      "script",
      { type: "application/ld+json" },
      JSON.stringify({
        "@context": "https://schema.org",
        "@graph": graph
      })
    ]);
  },

  themeConfig: {
    logo: "/logo.png",
    siteTitle: "num.zig",

    nav: [
      { text: "Home", link: "/" },
      { text: "Guide", link: "/guide/getting-started" },
      { text: "API Reference", link: "/api/" },
      { text: "Examples", link: "/guide/examples" },
      { text: "Benchmarks", link: "/guide/benchmarks" },
      { text: "Releases", link: "https://github.com/muhammad-fiaz/num.zig/releases" },
      {
        text: "Support",
        items: [
          { text: "💖 Sponsor", link: "https://github.com/sponsors/muhammad-fiaz" },
          { text: "☕ Donate", link: "https://pay.muhammadfiaz.com" },
        ],
      },
      { text: "GitHub", link: "https://github.com/muhammad-fiaz/num.zig" },
    ],

    sidebar: {
      "/guide/": [
        {
          text: "Getting Started",
          items: [
            { text: "Introduction", link: "/guide/introduction" },
            { text: "Getting Started", link: "/guide/getting-started" },
            { text: "Installation", link: "/guide/installation" },
            { text: "Architecture & SBO", link: "/guide/architecture" },
          ],
        },
        {
          text: "Core Concepts",
          items: [
            { text: "Array Creation", link: "/guide/array-creation" },
            { text: "DTypes & Promotion", link: "/guide/dtypes" },
            { text: "Indexing & Slicing", link: "/guide/indexing" },
            { text: "Broadcasting", link: "/guide/broadcasting" },
            { text: "Shape Manipulation", link: "/guide/manipulation" },
          ],
        },
        {
          text: "Numerical Operations",
          items: [
            { text: "Vectorized Math", link: "/guide/math" },
            { text: "Reductions & Accumulations", link: "/guide/reductions" },
            { text: "Linear Algebra", link: "/guide/linalg" },
            { text: "FFT & Spectral", link: "/guide/fft" },
            { text: "Statistics & Correlation", link: "/guide/statistics" },
            { text: "Random Distributions", link: "/guide/random" },
            { text: "Sorting & Set Operations", link: "/guide/sorting" },
            { text: "Polynomial Calculus", link: "/guide/polynomials" },
            { text: "Sparse Matrices & Solvers", link: "/guide/sparse" },
            { text: "CPU Parallel Execution", link: "/guide/parallel" },
            { text: "NZIG v1.0 & Text I/O", link: "/guide/io" },
          ],
        },
        {
          text: "Reference",
          items: [
            { text: "Runnable Examples", link: "/guide/examples" },
            { text: "Performance Benchmarks", link: "/guide/benchmarks" },
          ],
        },
      ],
      "/api/": [
        {
          text: "API Reference",
          items: [
            { text: "API Overview", link: "/api/" },
            { text: "Core Array & Memory", link: "/api/core" },
            { text: "DType & Promotion", link: "/api/dtype" },
            { text: "Elementwise Math", link: "/api/elementwise" },
            { text: "Comparisons & Logic", link: "/api/compare" },
            { text: "Reductions", link: "/api/reduce" },
            { text: "Shape Manipulation", link: "/api/manip" },
            { text: "Linear Algebra", link: "/api/linalg" },
            { text: "Fast Fourier Transform", link: "/api/fft" },
            { text: "Sparse Matrix", link: "/api/sparse" },
            { text: "Parallel Execution", link: "/api/parallel" },
            { text: "Random Distributions", link: "/api/random" },
            { text: "Statistics", link: "/api/stats" },
            { text: "Sorting & Searching", link: "/api/sort" },
            { text: "Polynomials", link: "/api/poly" },
            { text: "Serialization & I/O", link: "/api/io" },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/muhammad-fiaz/num.zig" },
    ],

    footer: {
      message: "Released under the MIT License.",
      copyright: "Copyright © 2026 Muhammad Fiaz",
    },

    search: {
      provider: "local",
    },

    editLink: {
      pattern: "https://github.com/muhammad-fiaz/num.zig/edit/main/docs/:path",
      text: "Edit this page on GitHub",
    },

    lastUpdated: {
      text: "Last updated",
      formatOptions: {
        dateStyle: "medium",
        timeStyle: "short",
      },
    },
  },
});
