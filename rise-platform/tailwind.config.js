module.exports = {
  purge: [],
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {
      colors: {
        theme: 'var(--theme-color)', // this allows `text-theme`, `bg-theme`, etc.
      },
    },
  },
  variants: {
    extend: {},
  },
  plugins: [],
}
