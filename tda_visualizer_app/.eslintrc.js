module.exports = {
  extends: ['react-app'],
  globals: {
    BigInt: 'readonly',
    FinalizationRegistry: 'readonly'
  },
  rules: {
    'no-undef': 'warn',
    'eqeqeq': 'warn',
    'no-new-object': 'warn',
    'no-array-constructor': 'warn'
  }
};
