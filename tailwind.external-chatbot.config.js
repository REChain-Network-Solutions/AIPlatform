import defaultConfig from './tailwind.config';

/** @type {import('tailwindcss').Config} */
export default {
	content: [
		'./resources/views/**/components/**/*.blade.php',
		'./app/Extensions/Chatbot/**/*.blade.php',
	],
	theme: {
		extend: {
			...defaultConfig.theme.extend,
			screens: {
				sm: '576px',
				md: '768px',
				lg: '992px',
				xl: '1170px',
				'2xl': '1170px',
			},
			borderRadius: {
				'3xl': '1.375rem'
			},
			spacing: {
				'4.5': '1.125rem'
			},
			boxShadow: {
				'xs': '0 1px 11px rgb(0 0 0 / 6%)',
				'sm': '0 3px 6px rgb(0 0 0 / 16%)',
				'lg': '0 15px 33px rgb(0 0 0 / 5%)',
				'xl': '0 20px 50px rgb(0 0 0 / 20%)',
				'2xl': '0 33px 44px rgb(0 0 0 / 12%)',
			},
			keyframes: {
				...defaultConfig.theme.extend.keyframes,
				'pulse-intense': {
					'0%, 100%': { opacity: 1, transform: 'scale(1)' },
					'50%': { opacity: 0.5, transform: 'scale(0.75)' },
				},
				'hue-rotate': {
					'0%': { filter: 'hue-rotate(0deg)' },
					'100%': { filter: 'hue-rotate(360deg)' },
				},
			},
			animation: {
				...defaultConfig.theme.extend.animation,
				'pulse-intense': 'pulse-intense 2s ease-in-out infinite',
				'hue-rotate': 'hue-rotate 1.9s linear infinite',
			},
		},
	},
	plugins: [
		require( '@tailwindcss/typography' ),
		require( 'tailwindcss-motion' )
	]
};
