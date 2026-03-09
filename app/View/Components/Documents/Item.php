<?php

namespace App\View\Components\Documents;

use App\Extensions\AiVideoPro\System\Models\UserFall;
use App\Models\Folders;
use App\Models\UserOpenai;
use Closure;
use Illuminate\Contracts\View\View;
use Illuminate\Database\Eloquent\Collection;
use Illuminate\Database\Eloquent\Model;
use Illuminate\View\Component;

class Item extends Component
{
    /**
     * Create a new component instance.
     */
    public function __construct(
        public UserOpenai|UserFall|Model $entry,
        public string $style = 'min',
        public string $trim = '50',
        public bool $hideFav = false,
        public null|Collection|Folders $folders = null,
    ) {}

    /**
     * Get the view / contents that represent the component.
     */
    public function render(): View|Closure|string
    {
        return view('components.documents.item');
    }
}
