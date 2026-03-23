<?php

namespace App\View\Components;

use App\Domains\Marketplace\Repositories\Contracts\ExtensionRepositoryInterface;
use Closure;
use Illuminate\Contracts\View\View;
use Illuminate\View\Component;

class DemoSwitcher extends Component
{
    public array $themes;

    /**
     * Featured themes shown in both demo switcher and dummy theme switcher.
     *
     * @return array<int, array{theme_type: string, name: string, slug: string, icon: string, price: int, extension: bool}>
     */
    public static function featuredThemes(string $themesType = 'All'): array
    {
        return [
            [
                'theme_type' => $themesType,
                'name'       => 'AI Chat Pro',
                'slug'       => 'aichatpro',
                'icon'       => 'http://liquidlabs.uk/market/assets/icons/ai-chat-pro.jpg',
                'price'      => 10,
                'extension'  => true,
            ],
            [
                'theme_type' => $themesType,
                'name'       => 'AI Image Pro',
                'slug'       => 'imagepro',
                'icon'       => 'http://liquidlabs.uk/market/assets/icons/imagepro.jpg',
                'price'      => 10,
                'extension'  => true,
            ],
        ];
    }

    /**
     * Create a new component instance.
     */
    public function __construct(
        protected ExtensionRepositoryInterface $inter,
        public string $themesType = 'Frontend'|'Dashboard'|'All',
    ) {
        $this->themes = array_merge(self::featuredThemes($themesType), $inter->themes());
    }

    /**
     * Get the view / contents that represent the component.
     */
    public function render(): View|Closure|string
    {
        return view('components.demo-switcher');
    }
}
