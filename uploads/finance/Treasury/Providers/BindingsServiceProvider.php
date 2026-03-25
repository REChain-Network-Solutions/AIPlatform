<?php
namespace Modules\Treasury\Providers;
use Illuminate\Support\ServiceProvider;
use Modules\Treasury\Contracts\LedgerPosterInterface;
use Modules\Treasury\Services\AccountingLedgerPoster;

class BindingsServiceProvider extends ServiceProvider
{
    public function register(): void
    {
        $this->app->bind(LedgerPosterInterface::class, function () {
            return new AccountingLedgerPoster();
        });
    }
}
