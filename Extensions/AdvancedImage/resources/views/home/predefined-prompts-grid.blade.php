@php
    $prompts = [
        [
            'title' => 'Wallpaper',
            'prompt' => 'Create a stunning high-resolution wallpaper featuring vibrant abstract art with dynamic shapes and colors.',
            'ai_model' => 'flux-pro',
        ],
        [
            'title' => 'ComicBook',
            'prompt' => 'Design a dynamic comic book cover with bold characters, action-packed poses, and a dramatic backdrop.',
            'ai_model' => 'flux-pro',
        ],
        [
            'title' => 'Architecture',
            'prompt' => 'Generate a futuristic architectural design showcasing innovative skyscrapers with eco-friendly elements.',
            'ai_model' => 'flux-pro',
        ],
        [
            'title' => 'Digital Illustration',
            'prompt' => 'Create a digital illustration of a serene fantasy landscape with glowing trees and a magical river.',
            'ai_model' => 'flux-pro',
        ],
        [
            'title' => 'Abstract Background',
            'prompt' => 'Design a modern abstract background with flowing gradients, geometric patterns, and subtle textures.',
            'ai_model' => 'flux-pro',
        ],
        [
            'title' => 'Product Mockups',
            'prompt' => 'Generate a realistic product mockup featuring sleek packaging and minimalistic design elements.',
            'ai_model' => 'flux-pro',
        ],
        [
            'title' => 'Pattern',
            'prompt' => 'Create a seamless pattern inspired by nature with floral motifs and earthy tones.',
            'ai_model' => 'flux-pro',
        ],
        [
            'title' => 'Storyboard',
            'prompt' => 'Design a storyboard for a short film featuring a suspenseful chase scene in a bustling city.',
            'ai_model' => 'flux-pro',
        ],
        [
            'title' => 'Movie Poster',
            'prompt' => 'Create a dramatic movie poster for a sci-fi thriller, with a mysterious protagonist and a glowing cityscape.',
            'ai_model' => 'flux-pro',
        ],
        [
            'title' => 'Album Cover',
            'prompt' => 'Design an album cover with a retro aesthetic featuring bold typography and vibrant neon colors.',
            'ai_model' => 'flux-pro',
        ],
        [
            'title' => 'Children’s Book',
            'prompt' => 'Illustrate a children’s book page showing a whimsical forest with friendly animals and colorful scenery.',
            'ai_model' => 'flux-pro',
        ],
        [
            'title' => 'Book Cover',
            'prompt' => 'Design an elegant book cover for a mystery novel with intricate patterns and a shadowy figure.',
            'ai_model' => 'flux-pro',
        ],
    ];
@endphp

<div class="lqd-adv-editor-predefined-prompts border-b pb-20 pt-9">

    <div class="lqd-adv-editor-predefined-prompts-grid flex flex-wrap gap-4">
        @foreach ($prompts as $prompt)
            <a
                class="lqd-adv-editor-predefined-prompts-grid-item inline-flex rounded-full bg-heading-foreground/5 px-4 py-2.5 text-2xs font-medium text-heading-foreground transition-all hover:scale-110 hover:bg-primary hover:text-primary-foreground hover:shadow-lg hover:shadow-black/5"
                data-prompt="{{ $prompt['prompt'] }}"
                data-ai-model="{{ $prompt['ai_model'] }}"
                href="#"
                @click.prevent='selectedTemplate = "{{ $prompt['title'] }}"; selectedTemplateDescription = ""; selectedPromptDescription = "{{ $prompt['prompt'] }}"; window.scrollTo({ top: 0, behavior: "smooth" });'
            >
                {{ $prompt['title'] }}
            </a>
        @endforeach
    </div>
</div>
