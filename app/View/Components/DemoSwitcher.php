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
     * Create a new component instance.
     */
    public function __construct(
        protected ExtensionRepositoryInterface $inter,
        public string $themesType = 'Frontend'|'Dashboard'|'All',
    ) {
        $this->themes = array_merge([[
            'theme_type' => $themesType,
            'name'       => 'AI Chat Pro',
            'slug'       => 'aichatpro',
            'icon'       => 'http://liquidlabs.uk/market/assets/icons/ai-chat-pro.jpg',
            'price'      => 10,
            'extension'  => true,
        ]], $inter->themes());
    }

    /**
     * Get the view / contents that represent the component.
     */
    public function render(): View|Closure|string
    {
        return view('components.demo-switcher');
    }
}
