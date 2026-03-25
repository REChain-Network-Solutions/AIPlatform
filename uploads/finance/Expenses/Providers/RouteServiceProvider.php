<?php
namespace Modules\Expenses\Providers; use Illuminate\Support\ServiceProvider;
class RouteServiceProvider extends ServiceProvider{public function boot():void{foreach(['web','api','ai'] as $f){$p=__DIR__.'/../Routes/'.$f.'.php';if(file_exists($p))$this->loadRoutesFrom($p);}}}