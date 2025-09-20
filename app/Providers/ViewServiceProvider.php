<?php

namespace App\Providers;

use App\Helpers\Classes\Helper;
use App\Helpers\Classes\TableSchema;
use App\Models\Frontend\FrontendSectionsStatus;
use App\Models\Frontend\FrontendSetting;
use App\Models\OpenAIGenerator;
use App\Models\Section\AdvancedFeaturesSection;
use App\Models\Section\BannerBottomText;
use App\Models\Section\ComparisonSectionItems;
use App\Models\Section\FeaturesMarquee;
use App\Models\Section\FooterItem;
use App\Models\Setting;
use App\Models\SettingTwo;
use App\View\Composers\PlanComposer;
use Illuminate\Pagination\Paginator;
use Illuminate\Support\Facades\View;
use Illuminate\Support\ServiceProvider;

class ViewServiceProvider extends ServiceProvider
{
    protected array $tables = [];

    protected ?Setting $settings = null;

    public function register(): void {}

    public function boot(): void
    {
        $this->sharedAppStatus();
        Paginator::useBootstrap();

        if (! Helper::dbConnectionStatus()) {
            return;
        }

        $this->tables = app('magicai_tables');

        if (! $this->hasTables(['migrations', 'settings'])) {
            return;
        }

        $this->shareSetting();
        $this->shareAiGenerator();
        $this->shareGoodForNow();

        View::composer(
            ['components.navbar.navbar', 'panel.layout.partials.menu'],
            PlanComposer::class
        );
    }

    protected function hasTables(array $requiredTables): bool
    {
        return collect($requiredTables)->every(fn ($table) => TableSchema::hasTable($table, $this->tables));
    }

    protected function shareGoodForNow(): void
    {
        $goodForNow = TableSchema::hasTable('settings_two', $this->tables)
            && $this->settings
            && Helper::settingTwo('liquid_license_type');

        View::share('good_for_now', $goodForNow);
    }

    protected function shareAiGenerator(): void
    {
        if (! TableSchema::hasTable('openai', $this->tables)) {
            return;
        }

        $generators = OpenAIGenerator::where('active', 1)->orderBy('title')->get();
        View::share('aiWriters', $generators);
    }

    protected function shareSetting(): void
    {
        if ($settings = Setting::getCache()) {
            $this->settings = $settings;
            View::share('setting', $settings);
        }

        $this->shareFrontendSettings();
        $this->shareOpenAiList();
        $this->shareSections();
        $this->shareCommissionSetting();
        $this->shareSettingsTwo();
    }

    protected function shareFrontendSettings(): void
    {
        $this->conditionallyShare('frontend_footer_settings', function () {
            $frontendSetting = FrontendSetting::getCache() ?? tap(new FrontendSetting)->save();
            View::share('fSetting', $frontendSetting);
        });

        $this->conditionallyShare('frontend_sections_statuses_titles', function () {
            $fSectSettings = FrontendSectionsStatus::getCache() ?? tap(new FrontendSectionsStatus)->save();
            View::share('fSectSettings', $fSectSettings);
        });
    }

    protected function shareOpenAiList(): void
    {
        if (TableSchema::hasTable('openai', $this->tables)) {
            View::share('openAiList', OpenAIGenerator::all());
        }
    }

    protected function shareSections(): void
    {
        $sectionTables = [
            'advanced_features_section' => fn () => View::share('advanced_features_section', AdvancedFeaturesSection::all()),
            'comparison_section_items'  => fn () => View::share('comparison_section_items', ComparisonSectionItems::all()),
            'features_marquees'         => fn () => $this->shareMarqueeItems(),
            'footer_items'              => fn () => View::share('footer_items', FooterItem::pluck('item')->toArray()),
            'banner_bottom_texts'       => fn () => $this->shareBannerBottomTexts(),
        ];

        foreach ($sectionTables as $table => $callback) {
            $this->conditionallyShare($table, $callback);
        }
    }

    protected function shareMarqueeItems(): void
    {
        $marquees = FeaturesMarquee::getCache(fn () => FeaturesMarquee::select('title', 'position')->get());
        View::share('top_marquee_items', $marquees->where('position', 'top')->pluck('title')->toArray());
        View::share('bottom_marquee_items', $marquees->where('position', 'bottom')->pluck('title')->toArray());
    }

    protected function shareBannerBottomTexts(): void
    {
        $texts = BannerBottomText::getCache(fn () => BannerBottomText::pluck('text')->toArray());
        View::share('banner_bottom_texts', $texts);
    }

    protected function shareCommissionSetting(): void
    {
        $this->conditionallyShare('app_settings', function () {
            View::share('is_onetime_commission', setting('onetime_commission', 0));
        });
    }

    protected function shareSettingsTwo(): void
    {
        $this->conditionallyShare('settings_two', function () {
            $settingsTwo = SettingTwo::getCache() ?? tap(new SettingTwo)->save();
            View::share('settings_two', $settingsTwo);
        });
    }

    protected function sharedAppStatus(): void
    {
        View::share('app_is_demo', Helper::appIsDemo());
        View::share('app_is_not_demo', Helper::appIsNotDemo());
    }

    protected function conditionallyShare(string $table, callable $callback): void
    {
        if (TableSchema::hasTable($table, $this->tables)) {
            $callback();
        }
    }
}
