<?php
namespace Modules\Expenses\Providers; use Illuminate\Support\ServiceProvider;
class AiRoutesServiceProvider extends ServiceProvider{public function boot():void{$this->loadRoutesFrom(__DIR__.'/../Routes/ai.php');}}