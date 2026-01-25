<?php

/**
 * Class MagicAI_Main_Settings
 *
 * Configure the plugin settings page.
 */
class MagicAI_Main_Settings {

	/**
	 * Instance
	 *
	 * @since 1.0.0
	 *
	 * @access private
	 * @static
	 *
	 * @var MagicAI_Main_Settings The single instance of the class.
	 */
	private static $_instance = null;

    /**
	 * Instance
	 *
	 * Ensures only one instance of the class is loaded or can be loaded.
	 *
	 * @since 1.0.0
	 *
	 * @access public
	 * @static
	 *
	 * @return MagicAI_Main_Settings An instance of the class.
	 */
	public static function instance() {

		if ( is_null( self::$_instance ) ) {
			self::$_instance = new self();
		}
		return self::$_instance;
    }

	/**
	 * Capability required by the user to access the My Plugin menu entry.
	 *
	 * @var string $capability
	 */
	private $capability = 'manage_options';

	/**
	 * Array of fields that should be displayed in the settings page.
	 *
	 * @var array $fields
	 */
	private $tabs;

	/**
	 * The Plugin Settings constructor.
	 */
	function __construct() {
		$this->tabs = [
			'openai' => [
				'label' => esc_html__('OpenAI', 'magicai-wp'),
				'sections' =>[
					'openai' => [
					   'label' => esc_html__('OpenAI Settings', 'magicai-wp'),
					   'fields' => [
							[
								'id' => 'openai_key',
								'label' => esc_html__('API Key', 'magicai-wp'),
								'description' => __('You can find your API key at <a href="https://platform.openai.com/account/api-keys" target="_blank">https://platform.openai.com/account/api-keys</a>', 'magicai-wp'),
								'type' => 'text',
							],
							[
								'id' => 'openai_model',
								'label' => esc_html__('Model','magicai-wp'),
								'description' => esc_html__('The OpenAI API is powered by a diverse set of models with different capabilities and price points.','magicai-wp'),
								'type' => 'select',
								'options' => [
									'gpt-4' => 'gpt-4',
									'gpt-4-32k' => 'gpt-4-32k',
									'gpt-4-0125-preview' => 'gpt-4-0125-preview',
									'gpt-3.5-turbo' => 'gpt-3.5-turbo',
									'gpt-3.5-turbo-0125' => 'gpt-3.5-turbo-0125',
									'gpt-3.5-turbo-16k' => 'gpt-3.5-turbo-16k',
									'gpt-3.5-turbo-instruct' => 'gpt-3.5-turbo-instruct (Similar capabilities as text-davinci-003)',
								],
								'default' => 'gpt-3.5-turbo-instruct',
							],
							[
								'id' => 'openai_tts_model',
								'label' => esc_html__('Text to Speech Model (TTS)','magicai-wp'),
								'description' => esc_html__('TTS is an AI model that converts text to natural sounding spoken text. We offer two different model variates, tts-1 is optimized for real time text to speech use cases and tts-1-hd is optimized for quality.','magicai-wp'),
								'type' => 'select',
								'options' => [
									'tts-1' => 'tts-1',
									'tts-1-hd' => 'tts-1-hd',
								],
								'default' => 'tts-1',
							],
							[
								'id' => 'openai_max_tokens',
								'label' => esc_html__('Max Tokens','magicai-wp'),
								'description' => esc_html__('It allows users to limit the length of the generated text to a specified number of tokens to ensure it fits within desired constraints or to manage response length.','magicai-wp'),
								'type' => 'number',
								'options' => [
									'min' => 50,
									'max' => 32000,
									'step' => 5
								],
								'default' => 300,
							],
							[
								'id' => 'openai_temperature',
								'label' => esc_html__('Temperature', 'magicai-wp'),
								'description' => esc_html__('This parameter used in OpenAI\'s language models like GPT-3.5 to control the randomness of generated text. Higher values (e.g., Premium) make the output more diverse and creative by introducing randomness, while lower values (e.g., Economic) make the output more focused and deterministic by reducing randomness.','magicai-wp'),
								'type' => 'select',
								'options' => [
									'0.25' => 'Economic',
									'0.5' => 'Average',
									'0.75' => 'Good',
									'1' => 'Premium',
								],
								'default' => '0.75',
							],
							[
								'id' => 'openai_frequency_penalty',
								'label' => esc_html__('Frequency Penalty', 'magicai-wp'),
								'description' => esc_html__('This parameter in OpenAI\'s language models that regulates the model\'s tendency to generate commonly used words or phrases, with higher values leading to more diverse and creative outputs by reducing the repetition of common words.', 'magicai-wp'),
								'type' => 'number',
								'options' => [
									'min' => 0,
									'max' => 1,
									'step' => 0.1
								],
								'default' => 0,
							],
							[
								'id' => 'openai_presence_penalty',
								'label' => esc_html__('Presence Penalty', 'magicai-wp'),
								'description' => esc_html__('This parameter in OpenAI\'s language models that controls the model\'s inclination to include or avoid certain specified words or phrases in its generated output, allowing users to influence the presence or absence of specific content in the text.', 'magicai-wp'),
								'type' => 'number',
								'options' => [
									'min' => 0,
									'max' => 1,
									'step' => 0.1
								],
								'default' => 0.6,
							],
							[
								'id' => 'openai_default_language',
								'label' => esc_html__('Generator Forms Default Language', 'magicai-wp'),
								'description' => esc_html__('Set the default language for forms generated by the OpenAI language model. This language will be applied as the initial language setting for new forms created through the generator.', 'magicai-wp'),
								'type' => 'select',
								'options' => [
									'ar-AE' => esc_html__('Arabic', 'magicai-wp'),
									'cmn-CN'=> esc_html__('Chinese (Mandarin)', 'magicai-wp'),
									'cs-CZ' => esc_html__('Czech (Czech Republic)', 'magicai-wp'),
									'da-DK' => esc_html__('Danish (Denmark)', 'magicai-wp'),
									'de-DE' => esc_html__('German (Germany)', 'magicai-wp'),
									'el-GR' => esc_html__('Greek (Greece)', 'magicai-wp'),
									'en-US' => esc_html__('English (USA)', 'magicai-wp'),
									'es-ES' => esc_html__('Spanish (Spain)', 'magicai-wp'),
									'et-EE' => esc_html__('Estonian (Estonia)', 'magicai-wp'),
									'fi-FI' => esc_html__('Finnish (Finland)', 'magicai-wp'),
									'fr-FR' => esc_html__('French (France)', 'magicai-wp'),
									'he-IL' => esc_html__('Hebrew (Israel)', 'magicai-wp'),
									'hi-IN' => esc_html__('Hindi (India)', 'magicai-wp'),
									'hr-HR' => esc_html__('Croatian (Croatia)', 'magicai-wp'),
									'hu-HU' => esc_html__('Hungarian (Hungary)', 'magicai-wp'),
									'id-ID' => esc_html__('Indonesian (Indonesia)', 'magicai-wp'),
									'is-IS' => esc_html__('Icelandic (Iceland)', 'magicai-wp'),
									'it-IT' => esc_html__('Italian (Italy)', 'magicai-wp'),
									'ja-JP' => esc_html__('Japanese (Japan)', 'magicai-wp'),
									'kk-KZ' => esc_html__('Kazakh (Kazakhistan)', 'magicai-wp'),
									'ko-KR' => esc_html__('Korean (South Korea)', 'magicai-wp'),
									'lt-LT' => esc_html__('Lithuanian (Lithuania)', 'magicai-wp'),
									'ms-MY' => esc_html__('Malay (Malaysia)', 'magicai-wp'),
									'nb-NO' => esc_html__('Norwegian (Norway)', 'magicai-wp'),
									'nl-NL' => esc_html__('Dutch (Netherlands)', 'magicai-wp'),
									'pl-PL' => esc_html__('Polish (Poland)', 'magicai-wp'),
									'pt-BR' => esc_html__('Portuguese (Brazil)', 'magicai-wp'),
									'pt-PT' => esc_html__('Portuguese (Portugal)', 'magicai-wp'),
									'ro-RO' => esc_html__('Romanian (Romania)', 'magicai-wp'),
									'ru-RU' => esc_html__('Russian (Russia)', 'magicai-wp'),
									'sl-SI' => esc_html__('Slovenian (Slovenia)', 'magicai-wp'),
									'sv-SE' => esc_html__('Swedish (Sweden)', 'magicai-wp'),
									'sw-KE' => esc_html__('Swahili (Kenya)', 'magicai-wp'),
									'tr-TR' => esc_html__('Turkish (Turkey)', 'magicai-wp'),
									'vi-VN' => esc_html__('Vietnamese (Vietnam)', 'magicai-wp'),
								],
								'default' => 'en-US',
							],
							[
								'id' => 'openai_default_tone',
								'label' => esc_html__('Generator Forms Default Tone', 'magicai-wp'),
								'description' => esc_html__('Set the default tone for forms generated by the OpenAI language model. This tone will be applied as the initial tone setting for new forms created through the generator.', 'magicai-wp'),
								'type' => 'select',
								'options' => [
									'Professional' => esc_html__('Professional', 'magicai-wp'),
									'Funny' => esc_html__('Funny', 'magicai-wp'),
									'Casual' => esc_html__('Casual', 'magicai-wp'),
									'Excited' => esc_html__('Excited', 'magicai-wp'),
									'Witty' => esc_html__('Witty', 'magicai-wp'),
									'Sarcastic' => esc_html__('Sarcastic', 'magicai-wp'),
									'Feminine' => esc_html__('Feminine', 'magicai-wp'),
									'Masculine' => esc_html__('Masculine', 'magicai-wp'),
									'Bold' => esc_html__('Bold', 'magicai-wp'),
									'Dramatic' => esc_html__('Dramatic', 'magicai-wp'),
									'Grumpy' => esc_html__('Grumpy', 'magicai-wp'),
									'Secretive' => esc_html__('Secretive', 'magicai-wp'),
								],
								'default' => 'Professional',
							],
					   ]
					],
					'dalle' => [
					   'label' => esc_html__('DALL-E Settings', 'magicai-wp'),
					   'fields' => [
							[
								'id' => 'dalle_model',
								'label' => esc_html__('Model','magicai-wp'),
								'description' => esc_html__('The OpenAI API is powered by a diverse set of models with different capabilities and price points.','magicai-wp'),
								'type' => 'select',
								'options' => [
									'dall-e-2' => 'DALL-E 2',
									'dall-e-3' => 'DALL-E 3',
								],
								'default' => 'dall-e-2',
							],
							[
								'id' => 'dalle_n',
								'label' => esc_html__('Number of Results', 'magicai-wp'),
								'description' => esc_html__('Number of results', 'magicai-wp'),
								'type' => 'number',
								'options' => [
									'min' => 1,
									'max' => 10,
									'step' => 1
								],
								'default' => 1,
							],
							[
								'id' => 'dalle_size',
								'label' => esc_html__('Image size', 'magicai-wp'),
								'description' => esc_html__('Generated image resolution.', 'magicai-wp'),
								'type' => 'select',
								'options' => [
									'256x256' => '256x256 (DALL-E 2)',
									'512x512' => '512x512 (DALL-E 2)',
									'1024x1024' => '1024x1024 (DALL-E 2 and 3)',
									'1024x1792' => '1024x1792 (DALL-E 3)',
									'1792x1024' => '1792x1024 (DALL-E 3)',
								],
								'default' => '1024x1024',
							],
					   ]
					],
					'openai_fine_tune' => [
					   'label' => esc_html__('Fine Tune', 'magicai-wp'),
					   'fields' => [
							[
								'id' => 'openai_fine_tune',
								'label' => esc_html__('Manage Fine Tunes', 'magicai-wp'),
								'description' => esc_html__('You receive an AI response supported by Google search results. By enabling this feature, you gain access to AI-powered web search capabilities, allowing you to retrieve valuable information and insights from a vast array of online sources directly through the application.', 'magicai-wp'),
								'type' => 'openai_fine_tune',
							],
					   ]
					],
					'openai_models' => [
					   'label' => esc_html__('OpenAI Models for Generators & Chat', 'magicai-wp'),
					   'fields' => [
							[
								'id' => 'openai_model_post_generator',
								'label' => esc_html__('Post Generator(s)', 'magicai-wp'),
								'description' => esc_html__('Select the desired model for the all Post generators.', 'magicai-wp'),
								'type' => 'openai_models',
							],
							[
								'id' => 'openai_model_custom_generator',
								'label' => esc_html__('Custom Generator', 'magicai-wp'),
								'description' => esc_html__('Select the desired model for the Custom Generator & Web Search', 'magicai-wp'),
								'type' => 'openai_models',
							],
							[
								'id' => 'openai_model_chat',
								'label' => esc_html__('Chat', 'magicai-wp'),
								'description' => esc_html__('Select the desired model for the Chat', 'magicai-wp'),
								'type' => 'openai_models',
							],
							[
								'id' => 'openai_model_chatbot',
								'label' => esc_html__('ChatBot', 'magicai-wp'),
								'description' => esc_html__('Select the desired model for the ChatBot', 'magicai-wp'),
								'type' => 'openai_models',
							],
							[
								'id' => 'openai_model_assistant',
								'label' => esc_html__('Assistant', 'magicai-wp'),
								'description' => esc_html__('Select the desired model for the Assistant (Block Editor & Classic Editor)', 'magicai-wp'),
								'type' => 'openai_models',
							],
					   ]
					],
				]
			],
			'assistant' => [
				'label' => esc_html__('Assistant', 'magicai-wp'),
				'sections' =>[
					'assistant_prompts' => [
					   'label' => esc_html__('Assistant', 'magicai-wp'),
					   'fields' => [
							[
								'id' => 'assistant_prompts',
								'label' => esc_html__('Prompts', 'magicai-wp'),
								'description' => esc_html__('Define prompts to Assistant for shortcuts!', 'magicai-wp'),
								'type' => 'sortable',
								'default' => '[{"icon":"<span class=\"dashicons dashicons-editor-alignleft\"></span>","prompt":"Summarize"},{"icon":"<span class=\"dashicons dashicons-editor-justify\"></span>","prompt":"Add more lines"},{"icon":"<span class=\"dashicons dashicons-editor-spellcheck\"></span>","prompt":"Fix Grammar"},{"icon":"<span class=\"dashicons dashicons-text\"></span>","prompt":"Create TL;DR"},{"icon":"<span class=\"dashicons dashicons-edit\"></span>","prompt":"Rewrite"},{"icon":"<span class=\"dashicons dashicons-translation\"></span>","prompt":"Translate to Spanish"},{"icon":"<span class=\"dashicons dashicons-editor-ul\"></span>","prompt":"Make a List"}]',
							],
					   ]
					],
				]
			],
			'prompts' => [
				'label' => esc_html__('Prompts', 'magicai-wp'),
				'sections' =>[
					'prompts' => [
					   'label' => esc_html__('Prompts', 'magicai-wp'),
					   'description' => esc_html__('Manage prompts for generators. If certain fields are left empty, the system will automatically employ default values to ensure seamless processing.', 'magicai-wp'),
					   'fields' => [
							[
								'id' => 'prompt_code_generator',
								'type' => 'text',
								'label' => esc_html__('Code Generator', 'magicai-wp'),
								'description' => esc_html__('Default: You are a [code_language] expert. Write code about [code_request]', 'magicai-wp'),
								'default' => 'You are a [code_language] expert. Write code about [code_request]'
							],
							[
								'id' => 'prompt_product_generator_title',
								'type' => 'text',
								'label' => esc_html__('Product Generator (Title)', 'magicai-wp'),
								'description' => esc_html__('Default: write one product title about [title]. The tone of voice should [tone] in [language] language', 'magicai-wp'),
								'default' => 'write one product title about [title]. The tone of voice should [tone] in [language] language'
							],
							[
								'id' => 'prompt_product_generator_content',
								'type' => 'text',
								'label' => esc_html__('Product Generator (Content)', 'magicai-wp'),
								'description' => esc_html__('Default: write product content with html tags(h2,p,strong,etc.) about [title]. The tone of voice should [tone] in [language] language. And maximum lenght [maximum_lenght]', 'magicai-wp'),
								'default' => 'write product content with html tags(h2,p,strong,etc.) about [title]. The tone of voice should [tone] in [language] language. And maximum lenght [maximum_lenght]'
							],
							[
								'id' => 'prompt_product_generator_tag',
								'type' => 'text',
								'label' => esc_html__('Product Generator (Tag)', 'magicai-wp'),
								'description' => esc_html__('Default: write 5 tags about (return only comma separated): about [title]. The tone of voice should [tone] in [language] language.', 'magicai-wp'),
								'default' => 'write 5 tags about (return only comma separated): about [title]. The tone of voice should [tone] in [language] language.'
							],
							[
								'id' => 'prompt_post_generator_title',
								'type' => 'text',
								'label' => esc_html__('Post Generator (Title)', 'magicai-wp'),
								'description' => esc_html__('Default: write one blog post title about [title]. The tone of voice should [tone] in [language] language', 'magicai-wp'),
								'default' => 'write one blog post title about [title]. The tone of voice should [tone] in [language] language'
							],
							[
								'id' => 'prompt_post_generator_content',
								'type' => 'text',
								'label' => esc_html__('Post Generator (Content)', 'magicai-wp'),
								'description' => esc_html__('Default: write blog post with html tags(h2,p,strong,etc.) about [title][tag]. The tone of voice should [tone] in [language] language. And maximum lenght [maximum_lenght]', 'magicai-wp'),
								'default' => 'write blog post with html tags(h2,p,strong,etc.) about [title][tag]. The tone of voice should [tone] in [language] language. And maximum lenght [maximum_lenght]'
							],
							[
								'id' => 'prompt_post_generator_tag',
								'type' => 'text',
								'label' => esc_html__('Post Generator (Tag)', 'magicai-wp'),
								'description' => esc_html__('Default: write 5 tags about (return only comma separated): about [title][tag]. The tone of voice should [tone] in [language] language.', 'magicai-wp'),
								'default' => 'write 5 tags about (return only comma separated): about [title][tag]. The tone of voice should [tone] in [language] language.'
							],
							[
								'id' => 'prompt_post_generator_tag_prefix',
								'type' => 'text',
								'label' => esc_html__('Post Generator (Tag Prefix)', 'magicai-wp'),
								'description' => esc_html__('Default: and related post tags [tag]', 'magicai-wp'),
								'default' => 'and related post tags [tag]'
							],
							[
								'id' => 'prompt_youtube_blog',
								'type' => 'text',
								'label' => esc_html__('Youtube - Prepare Blog Post', 'magicai-wp'),
								'description' => esc_html__('Default: You are blog writer. Turn the given transcript text into a blog post in and translate to [language] language. Group the content and create a subheading (witth HTML-h2) for each group. Content:', 'magicai-wp'),
								'default' => 'You are blog writer. Turn the given transcript text into a blog post in and translate to [language] language. Group the content and create a subheading (witth HTML-h2) for each group. Content:'
							],
							[
								'id' => 'prompt_youtube_short',
								'type' => 'text',
								'label' => esc_html__('Youtube - Main Idea', 'magicai-wp'),
								'description' => esc_html__('Default: You are transcript editor. Make sense of the given content and explain the main idea. Your result should be in [language] language. Content:', 'magicai-wp'),
								'default' => 'You are transcript editor. Make sense of the given content and explain the main idea. Your result should be in [language] language. Content:'
							],
							[
								'id' => 'prompt_youtube_list',
								'type' => 'text',
								'label' => esc_html__('Youtube - Create List', 'magicai-wp'),
								'description' => esc_html__('Default: You are transcript editor. Make sense of the given content and make a list main ideas. Your result should be in [language] language. Content:', 'magicai-wp'),
								'default' => 'You are transcript editor. Make sense of the given content and make a list main ideas. Your result should be in [language] language. Content:'
							],
							[
								'id' => 'prompt_youtube_tldr',
								'type' => 'text',
								'label' => esc_html__('Youtube - Create TLDR', 'magicai-wp'),
								'description' => esc_html__('Default: You are transcript editor. Make short TLDR. Your result should be in [language] language. Content:', 'magicai-wp'),
								'default' => 'You are transcript editor. Make short TLDR. Your result should be in [language] language. Content:'
							],
							[
								'id' => 'prompt_youtube_pros_cons',
								'type' => 'text',
								'label' => esc_html__('Youtube - Prepare Pros and Cons', 'magicai-wp'),
								'description' => esc_html__('Default: You are transcript editor. Make short pros and cons. Your result should be in [language] language. Content:', 'magicai-wp'),
								'default' => 'You are transcript editor. Make short pros and cons. Your result should be in [language] language. Content:'
							],
							[
								'id' => 'prompt_rss_generator_content',
								'type' => 'text',
								'label' => esc_html__('RSS Post Generator (Content)', 'magicai-wp'),
								'description' => esc_html__('Default: write blog post with html tags(h2,p,strong,etc.) about [title]. The tone of voice should [tone] in [language] language. And lenght should [maximum_lenght] word', 'magicai-wp'),
								'default' => 'write blog post with html tags(h2,p,strong,etc.) about [title]. The tone of voice should [tone] in [language] language. And lenght should [maximum_lenght] word'
							],
							[
								'id' => 'prompt_rss_generator_tags',
								'type' => 'text',
								'label' => esc_html__('RSS Post Generator (Tags)', 'magicai-wp'),
								'description' => esc_html__('Default: write 5 tags about (return only comma separated): about [title]. The tone of voice should [tone] in [language] language.', 'magicai-wp'),
								'default' => 'write 5 tags about (return only comma separated): about [title]. The tone of voice should [tone] in [language] language.'
							],
					   ]
					],
				]
			],
			'stable_diffusion' => [
				'label' => esc_html__('Stable Diffusion', 'magicai-wp'),
				'sections' =>[
					'stable_diffusion' => [
					   'label' => esc_html__('Stable Diffusion Settings', 'magicai-wp'),
					   'fields' => [
							[
								'id' => 'sd_key',
								'label' => esc_html__('API Key', 'magicai-wp'),
								'description' => __('You can find your API key at <a href="https://platform.stability.ai/docs/getting-started/authentication" target="_blank">https://stability.ai/</a>', 'magicai-wp'),
								'type' => 'text',
							],
							[
								'id' => 'sd_model',
								'label' => esc_html__('Model', 'magicai-wp'),
								'description' => esc_html__('The Stable Diffusion API is powered by a diverse set of models with different capabilities and price points.', 'magicai-wp'),
								'type' => 'select',
								'options' => [
									'stable-diffusion-v1.5' => 'Stable Diffusion 1.5',
									'stable-diffusion-512-v2-1' => 'Stable Diffusion 2.1',
									'stable-diffusion-xl-beta-v2-2-2' => 'Stable Diffusion XL 0.8',
									'stable-diffusion-xl-1024-v0-9' => 'Stable Diffusion XL 0.9',
									'stable-diffusion-xl-1024-v1-0' => 'Stable Diffusion XL 1.0',
								],
								'default' => 'stable-diffusion-512-v2-1',
							],
							[
								'id' => 'sd_n',
								'label' => esc_html__('Number of Results', 'magicai-wp'),
								'description' => esc_html__('Number of results', 'magicai-wp'),
								'type' => 'number',
								'options' => [
									'min' => 1,
									'max' => 10,
									'step' => 1
								],
								'default' => 1,
							],
							[
								'id' => 'sd_size',
								'label' => esc_html__('Image size', 'magicai-wp'),
								'description' => esc_html__('Generated image resolution.', 'magicai-wp'),
								'type' => 'select',
								'options' => [
									'256' => '256x256',
									'512' => '512x512',
									'1024' => '1024x1024',
								],
								'default' => '512',
							],
					   ]
					],
				]
			],
			'google_cloud' => [
				'label' => esc_html__('Google (GCS)', 'magicai-wp'),
				'sections' =>[
					'google_cloud' => [
					   'label' => esc_html__('Google Cloud Services Settings', 'magicai-wp'),
					   'fields' => [
							[
								'id' => 'gcs_file',
								'label' => esc_html__('GCS File (JSON) path', 'magicai-wp'),
								'description' => __('You can find your API key at <a href="https://cloud.google.com" target="_blank">https://cloud.google.com</a>', 'magicai-wp'),
								'type' => 'text',
							],
							[
								'id' => 'gcs_name',
								'label' => esc_html__('GCS Project Name', 'magicai-wp'),
								'description' => '',
								'type' => 'text',
							],
					   ]
					],
					'google_custom_search' => [
					   'label' => esc_html__('Custom Search JSON API', 'magicai-wp'),
					   'description' => sprintf( '<div class="magicai-deprecated"><span>%1$s</span>%2$s</div>', __('Deprecated', 'magicai-wp'), __('This option deprecated with v1.2! Use Additional APIs > Serper API instead.', 'magicai-wp') ),
					   'fields' => [
							[
								'id' => 'google_search_api',
								'label' => esc_html__('API Key', 'magicai-wp'),
								'description' => __('You can find your API key at <a href="https://developers.google.com/custom-search/v1/overview" target="_blank">https://developers.google.com/custom-search/v1/overview</a>', 'magicai-wp'),
								'type' => 'text',
							],
							[
								'id' => 'google_search_cx',
								'label' => esc_html__('Search Engine ID', 'magicai-wp'),
								'description' => __('The identifier of the Programmable Search Engine. Details: <a href="https://cse.google.com" target="_blank">https://cse.google.com</a>', 'magicai-wp'),
								'type' => 'text',
							],
					   ]
					],
				]
			],
			'additional_apis' => [
				'label' => esc_html__('Additional APIs', 'magicai-wp'),
				'sections' =>[
					'storage' => [
					   'label' => esc_html__('Storage', 'magicai-wp'),
					   'fields' => [
							[
								'id' => 'storage',
								'label' => esc_html__('Storage', 'magicai-wp'),
								'description' => esc_html__('Select storage type.', 'magicai-wp'),
								'type' => 'select',
								'options' => [
									'wp' => esc_html__('WordPress', 'magicai-wp'),
									's3' => esc_html__('Amazon S3', 'magicai-wp'),
								],
								'default' => 'wp',
							],
					   ]
					],
					'serper' => [
					   'label' => esc_html__('Serper', 'magicai-wp'),
					   'fields' => [
							[
								'id' => 'serper_api_key',
								'label' => esc_html__('Serper API Key', 'magicai-wp'),
								'description' => __('You can find your API key at <a href="https://serper.dev" target="_blank">https://serper.dev</a>', 'magicai-wp'),
								'type' => 'text',
							],
					   ]
					],
					'unsplash' => [
					   'label' => esc_html__('Unsplash', 'magicai-wp'),
					   'fields' => [
							[
								'id' => 'unsplash_api_key',
								'label' => esc_html__('Unsplash API Key', 'magicai-wp'),
								'description' => __('You can find your API key at <a href="https://unsplash.com/oauth/applications" target="_blank">https://unsplash.com/oauth/applications</a>', 'magicai-wp'),
								'type' => 'text',
							],
					   ]
					],
					'amazon' => [
					   'label' => esc_html__('Amazon S3'),
					   'fields' => [
							[
								'id' => 'aws_access_key_id',
								'label' => esc_html__('AWS Access Key ID', 'magicai-wp'),
								'type' => 'text',
							],
							[
								'id' => 'aws_secret_access_key',
								'label' => esc_html__('AWS Secret Access Key', 'magicai-wp'),
								'type' => 'text',
							],
							[
								'id' => 'aws_default_region',
								'label' => esc_html__('AWS Default Region', 'magicai-wp'),
								'type' => 'text',
							],
							[
								'id' => 'aws_bucket',
								'label' => esc_html__('AWS Bucket', 'magicai-wp'),
								'type' => 'text',
							],
							[
								'id' => 'aws_bucket',
								'label' => esc_html__('AWS Bucket', 'magicai-wp'),
								'type' => 'text',
							],
					   ]
					],
				]
			],
		];
		add_action( 'admin_init', [$this, 'settings_init'] );
	}

	/**
	 * Register the settings and all fields.
	 */
	function settings_init() : void {

		// Register a new setting this page.
		register_setting( 'magicai_settings', 'magicai_options' );

        foreach( $this->tabs as $tab ) {

            /* Register All The Fields. */
            foreach( $tab['sections'] as $section_id => $section ) {

                $full_section_id = 'magicai_' . $section_id . '_section';

                // Register a new section.
                add_settings_section(
                    $full_section_id,
                	$section['label'],
                    [$this, 'render_section'],
                    'magicai_settings'
                );

                foreach ( $section['fields'] as $field ) {
                    // Register a new field in the main section.
                    add_settings_field(
                        $field['id'], /* ID for the field. Only used internally. To set the HTML ID attribute, use $args['label_for']. */
                        $field['label'], /* Label for the field. */
                        [$this, 'render_field'], /* The name of the callback function. */
                        'magicai_settings', /* The menu page on which to display this field. */
                        $full_section_id, /* The section of the settings page in which to show the box. */
                        [
                            'label_for' => $field['id'], /* The ID of the field. */
                            'class' => 'wporg_row', /* The class of the field. */
                            'field' => $field, /* Custom data for the field. */
                        ]
                    );
                }

            }

        }
	}

	/**
	 * Render the settings page.
	 */
	function render_options_page() : void {

		// check user capabilities
		if ( ! current_user_can( $this->capability ) ) {
			return;
		}

		// add error/update messages

		// check if the user have submitted the settings
		// WordPress will add the "settings-updated" $_GET parameter to the url
		if ( isset( $_GET['settings-updated'] ) ) {
			// add settings saved message with the class of "updated"
			add_settings_error( 'magicai_messages', 'magicai_message', esc_html__( 'Settings Saved', 'magicai-wp' ), 'updated' );
		}

		// show error/update messages
		settings_errors( 'magicai_messages' );
		?>
		<div class="wrap">
			<h1><?php echo esc_html( get_admin_page_title() ); ?></h1>
			<p><?php esc_html_e( 'Manage MagicAI settings.', 'magicai-wp' ); ?></p>

            <div id="magicai-settings-tabs-wrapper" class="nav-tab-wrapper">
				<?php
				$first = true;
				foreach ( $this->tabs as $tab_id => $tab ) {

					$active_class = '';

					if ( $first ) {
						$active_class = ' nav-tab-active';
						$first = false;
					}

					$sanitized_tab_id = esc_attr( $tab_id );
					$sanitized_tab_label = esc_html( $tab['label'] );

					// PHPCS - Escaped the relevant strings above.
					echo "<a id='magicai-settings-tab-{$sanitized_tab_id}' class='nav-tab{$active_class}' href='#tab-{$sanitized_tab_id}'>{$sanitized_tab_label}</a>"; // phpcs:ignore WordPress.Security.EscapeOutput.OutputNotEscaped
				}
				?>
			</div>

			<form action="options.php" method="post" class="magicai-form">
				<?php
				/* output security fields for the registered setting "wporg" */
				settings_fields( 'magicai_settings' );
				
				$first = true;
                foreach ( $this->tabs as $tab_id => $tab ) {
					$active_class = '';

					if ( $first ) {
						$active_class = ' active';
						$first = false;
					}

					$sanitized_tab_id = esc_attr( $tab_id );

					// PHPCS - $active_class is a non-dynamic string and $sanitized_tab_id is escaped above.
					echo "<div id='tab-{$sanitized_tab_id}' class='magicai-settings-form-page{$active_class}'>"; // phpcs:ignore WordPress.Security.EscapeOutput.OutputNotEscaped

					foreach ( $tab['sections'] as $section_id => $section ) {
						$full_section_id = 'magicai_' . $section_id . '_section';

						if ( ! empty( $section['label'] ) ) {
							echo '<h2>' . esc_html( $section['label'] ) . '</h2>';
						}
					
						if ( ! empty( $section['description'] ) ) {
							echo $section['description'];
						}

						if ( ! empty( $section['callback'] ) ) {
							$section['callback']();
						}

						echo '<table class="form-table">';

						do_settings_fields( 'magicai_settings', $full_section_id );

						echo '</table>';
					}

					echo '</div>';
				}

                submit_button();
				?>
			</form>
		</div>
		<?php
	}

	/**
	 * Render a settings field.
	 *
	 * @param array $args Args to configure the field.
	 */
	function render_field( array $args ) : void {

		$field = $args['field'];

		// Get the value of the setting we've registered with register_setting()
		$default = isset( $field['default'] ) ? $field['default'] : '';
		$options = get_option( 'magicai_options', [ $field['id'] => $default ] );

		switch ( $field['type'] ) {

			case "text": {
				?>
				<div class="form-field">
				<input
					type="text"
					id="<?php echo esc_attr( $field['id'] ); ?>"
					name="magicai_options[<?php echo esc_attr( $field['id'] ); ?>]"
					value="<?php echo isset( $options[ $field['id'] ] ) ? esc_attr( $options[ $field['id'] ] ) : ''; ?>"
				>
				<p class="description">
					<?php echo $field['description']; ?> 
				</p>
				</div>
				<?php
				break;
			}
			
			case "number": {
				?>
				<div class="form-field">
				<input
					type="number"
					id="<?php echo esc_attr( $field['id'] ); ?>"
					name="magicai_options[<?php echo esc_attr( $field['id'] ); ?>]"
					min="<?php echo esc_attr( $field['options']['min'] ); ?>"
					max="<?php echo esc_attr( $field['options']['max'] ); ?>"
					step="<?php echo esc_attr( $field['options']['step'] ); ?>"
					value="<?php echo isset( $options[ $field['id'] ] ) ? esc_attr( $options[ $field['id'] ] ) : ''; ?>"
				>
				<p class="description">
					<?php echo $field['description']; ?> 
				</p>
				</div>
				<?php
				break;
			}

			case "checkbox": {
				?>
				<div class="form-field">
				<input
					type="checkbox"
					id="<?php echo esc_attr( $field['id'] ); ?>"
					name="magicai_options[<?php echo esc_attr( $field['id'] ); ?>]"
					value="1"
					<?php echo isset( $options[ $field['id'] ] ) ? ( checked( $options[ $field['id'] ], 1, false ) ) : ( '' ); ?>
				>
				<p class="description">
					<?php echo esc_html( $field['description'] ); ?>
				</p>
				</div>
				<?php
				break;
			}

			case "textarea": {
				?>
				<div class="form-field">
				<textarea
					id="<?php echo esc_attr( $field['id'] ); ?>"
					name="magicai_options[<?php echo esc_attr( $field['id'] ); ?>]"
				><?php echo isset( $options[ $field['id'] ] ) ? esc_attr( $options[ $field['id'] ] ) : ''; ?></textarea>
				<p class="description">
					<?php echo esc_html( $field['description'] ); ?>
				</p>
				</div>
				<?php
				break;
			}

			case "select": {
				?>
				<div class="form-field">
				<select
					id="<?php echo esc_attr( $field['id'] ); ?>"
					name="magicai_options[<?php echo esc_attr( $field['id'] ); ?>]"
				>
					<?php foreach( $field['options'] as $key => $option ) { ?>
						<option value="<?php echo $key; ?>" 
							<?php echo isset( $options[ $field['id'] ] ) ? ( selected( $options[ $field['id'] ], $key, false ) ) : ( '' ); ?>
						>
							<?php echo $option; ?>
						</option>
					<?php } ?>
				</select>
				<p class="description">
					<?php echo esc_html( $field['description'] ); ?>
				</p>
				</div>
				<?php
				break;
			}

			case "password": {
				?>
				<div class="form-field">
				<input
					type="password"
					id="<?php echo esc_attr( $field['id'] ); ?>"
					name="magicai_options[<?php echo esc_attr( $field['id'] ); ?>]"
					value="<?php echo isset( $options[ $field['id'] ] ) ? esc_attr( $options[ $field['id'] ] ) : ''; ?>"
				>
				<p class="description">
					<?php echo esc_html( $field['description'] ); ?>
				</p>
				</div>
				<?php
				break;
			}

			case "wysiwyg": {
				wp_editor(
					isset( $options[ $field['id'] ] ) ? $options[ $field['id'] ] : '',
					$field['id'],
					array(
						'textarea_name' => 'magicai_options[' . $field['id'] . ']',
						'textarea_rows' => 5,
					)
				);
				break;
			}

			case "email": {
				?>
				<div class="form-field">
				<input
					type="email"
					id="<?php echo esc_attr( $field['id'] ); ?>"
					name="magicai_options[<?php echo esc_attr( $field['id'] ); ?>]"
					value="<?php echo isset( $options[ $field['id'] ] ) ? esc_attr( $options[ $field['id'] ] ) : ''; ?>"
				>
				<p class="description">
					<?php echo esc_html( $field['description'] ); ?>
				</p>
				</div>
				<?php
				break;
			}

			case "url": {
				?>
				<div class="form-field">
				<input
					type="url"
					id="<?php echo esc_attr( $field['id'] ); ?>"
					name="magicai_options[<?php echo esc_attr( $field['id'] ); ?>]"
					value="<?php echo isset( $options[ $field['id'] ] ) ? esc_attr( $options[ $field['id'] ] ) : ''; ?>"
				>
				<p class="description">
					<?php echo esc_html( $field['description'] ); ?>
				</p>
				</div>
				<?php
				break;
			}

			case "color": {
				?>
				<div class="form-field">
				<input
					type="color"
					id="<?php echo esc_attr( $field['id'] ); ?>"
					name="magicai_options[<?php echo esc_attr( $field['id'] ); ?>]"
					value="<?php echo isset( $options[ $field['id'] ] ) ? esc_attr( $options[ $field['id'] ] ) : ''; ?>"
				>
				<p class="description">
					<?php echo esc_html( $field['description'] ); ?>
				</p>
				</div>
				<?php
				break;
			}

			case "date": {
				?>
				<div class="form-field">
				<input
					type="date"
					id="<?php echo esc_attr( $field['id'] ); ?>"
					name="magicai_options[<?php echo esc_attr( $field['id'] ); ?>]"
					value="<?php echo isset( $options[ $field['id'] ] ) ? esc_attr( $options[ $field['id'] ] ) : ''; ?>"
				>
				<p class="description">
					<?php echo esc_html( $field['description'] ); ?>
				</p>
				</div>
				<?php
				break;
			}

			case "openai_fine_tune": {
				?>
					<button class="magicai-btn magicai-add--fine-tune" type="button" style="margin-bottom:12px"><?php esc_html_e( 'Add New', 'magicai-wp' ); ?></button>
					<table class="wp-list-table widefat striped table-view-list magicai-fine-tune-table">
						<thead>
							<tr>
								<th><?php esc_html_e( 'Custom Name', 'magicai-wp' ); ?></th>
								<th><?php esc_html_e( 'File ID', 'magicai-wp' ); ?></th>
								<th><?php esc_html_e( 'Bytes', 'magicai-wp' ); ?></th>
								<th><?php esc_html_e( 'Base Model', 'magicai-wp' ); ?></th>
								<th><?php esc_html_e( 'Fine Tuned Model', 'magicai-wp' ); ?></th>
								<th><?php esc_html_e( 'Status', 'magicai-wp' ); ?></th>
								<th><?php esc_html_e( 'Actions', 'magicai-wp' ); ?></th>
							</tr>
						</thead>
						<tbody>
							<?php MagicAI_Actions::instance()->get_fine_tune_table_row(); ?>
						</tbody>
					</table>
				<?php
				break;
			}

			case "openai_models": {
				$models = MagicAI_Actions::instance()->get_fine_tune_models();
				?>
				<div class="form-field">
				<select
					id="<?php echo esc_attr( $field['id'] ); ?>"
					name="magicai_options[<?php echo esc_attr( $field['id'] ); ?>]"
				>
					<?php foreach( $models as $key => $option ) { ?>
						<option value="<?php echo $key; ?>" 
							<?php echo isset( $options[ $field['id'] ] ) ? ( selected( $options[ $field['id'] ], $key, false ) ) : ( '' ); ?>
						>
							<?php echo $option; ?>
						</option>
					<?php } ?>
				</select>
				<p class="description">
					<?php echo esc_html( $field['description'] ); ?>
				</p>
				</div>
				<?php
				break;
			}

			case "sortable": {
				?>
				<button class="magicai-btn magicai-add--assistant-prompt" type="button" style="margin-bottom:12px"><?php esc_html_e( 'Add New', 'magicai-wp' ); ?></button>

				<div id="sortable-items" class="magicai-settings--sortable-items">
				<?php 
				if ( isset( $options[ $field['id'] ] ) && $options[ $field['id'] ] ) {
					$sortable_items = json_decode($options[ $field['id'] ], true); 
					foreach ( $sortable_items as $item ) {
						?>
							<div class="magicai-settings--sortable-item">
								<div class="form-field">
									<label for="icon">Icon (Dashicons or SVG)</label>
									<input type="text" data-field="icon" name="icon" value="<?php echo esc_attr( $item['icon'] ); ?>" placeholder="<?php echo esc_attr( 'Dashicons or SVG'); ?>">
								</div>
								<div class="form-field w-100">
									<label for="icon">Prompt</label>
									<input type="text" data-field="prompt" name="prompt" value="<?php echo esc_attr( $item['prompt'] ); ?>" placeholder="<?php echo esc_attr( 'make shorter this paragraph' ); ?>">
								</div>
								<div class="action">
									<div class="handle"><span class="dashicons dashicons-move"></span></div>
									<div class="remove"><span class="dashicons dashicons-trash"></span></div>
								</div>
							</div>
						<?php 
					}
				}
				?>
				</div>

				<input
					type="hidden"
					class="datas"
					id="<?php echo esc_attr( $field['id'] ); ?>"
					name="magicai_options[<?php echo esc_attr( $field['id'] ); ?>]"
					value="<?php echo isset( $options[ $field['id'] ] ) ? esc_attr( $options[ $field['id'] ] ) : ''; ?>"
				>

				<p class="description">
					<?php echo $field['description']; ?> 
				</p>
				<?php
				break;
			}

		}
	}


	/**
	 * Render a section on a page, with an ID and a text label.
	 *
	 * @since 1.0.0
	 *
	 * @param array $args {
	 *     An array of parameters for the section.
	 *
	 *     @type string $id The ID of the section.
	 * }
	 */
	function render_section( array $args ) : void {
		?>
		<p id="<?php echo esc_attr( $args['id'] ); ?>"><?php esc_html_e( 'Section', 'magicai-wp' ); ?></p>
		<?php
	}

}

MagicAI_Main_Settings::instance();
