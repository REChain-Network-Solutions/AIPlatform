<?php

namespace App\Http;

use App\Domains\Marketplace\Http\Middleware\NewExtensionInstalled;
use App\Http\Middleware\AdminPermissionMiddleware;
use App\Http\Middleware\Authenticate;
use App\Http\Middleware\CheckInstallation;
use App\Http\Middleware\Custom;
use App\Http\Middleware\Custom\LocaleMiddleware;
use App\Http\Middleware\Custom\ThemeMiddleware;
use App\Http\Middleware\DemoCheckMiddleware;
use App\Http\Middleware\EncryptCookies;
use App\Http\Middleware\PreventRequestsDuringMaintenance;
use App\Http\Middleware\RedirectIfAuthenticated;
use App\Http\Middleware\RefererMiddleware;
use App\Http\Middleware\SentryContextMiddleware;
use App\Http\Middleware\SurveyMiddleware;
use App\Http\Middleware\TrimStrings;
use App\Http\Middleware\TrustProxies;
use App\Http\Middleware\UpdateUserActivity;
use App\Http\Middleware\ValidateSignature;
use App\Http\Middleware\VerifyCsrfToken;
use Illuminate\Auth\Middleware\AuthenticateWithBasicAuth;
use Illuminate\Auth\Middleware\Authorize;
use Illuminate\Auth\Middleware\EnsureEmailIsVerified;
use Illuminate\Auth\Middleware\RequirePassword;
use Illuminate\Cookie\Middleware\AddQueuedCookiesToResponse;
use Illuminate\Foundation\Http\Kernel as HttpKernel;
use Illuminate\Foundation\Http\Middleware\ConvertEmptyStringsToNull;
use Illuminate\Foundation\Http\Middleware\ValidatePostSize;
use Illuminate\Http\Middleware\HandleCors;
use Illuminate\Http\Middleware\SetCacheHeaders;
use Illuminate\Routing\Middleware\SubstituteBindings;
use Illuminate\Routing\Middleware\ThrottleRequests;
use Illuminate\Session\Middleware\AuthenticateSession;
use Illuminate\Session\Middleware\StartSession;
use Illuminate\View\Middleware\ShareErrorsFromSession;
use Mcamara\LaravelLocalization\Middleware\LaravelLocalizationRedirectFilter;
use Mcamara\LaravelLocalization\Middleware\LaravelLocalizationRoutes;
use Mcamara\LaravelLocalization\Middleware\LaravelLocalizationViewPath;
use Mcamara\LaravelLocalization\Middleware\LocaleCookieRedirect;
use Mcamara\LaravelLocalization\Middleware\LocaleSessionRedirect;
use RachidLaasri\LaravelInstaller\Middleware\ApplicationCheckLicense;
use RachidLaasri\LaravelInstaller\Middleware\ApplicationStatus;

class Kernel extends HttpKernel
{
    /**
     * The application's global HTTP middleware stack.
     *
     * These middleware are run during every request to your application.
     *
     * @var array<int, class-string|string>
     */
    protected $middleware = [
        // \App\Http\Middleware\TrustHosts::class,
        TrustProxies::class,
        RefererMiddleware::class,
        HandleCors::class,
        PreventRequestsDuringMaintenance::class,
        ValidatePostSize::class,
        TrimStrings::class,
        ConvertEmptyStringsToNull::class,
    ];

    /**
     * The application's route middleware groups.
     *
     * @var array<string, array<int, class-string|string>>
     */
    protected $middlewareGroups = [
        'web' => [
            EncryptCookies::class,
            AddQueuedCookiesToResponse::class,
            StartSession::class,
            ShareErrorsFromSession::class,
            ApplicationCheckLicense::class,
            ApplicationStatus::class,
            VerifyCsrfToken::class,
            SubstituteBindings::class,
            LocaleMiddleware::class,
            ThemeMiddleware::class,
        ],

        'api' => [
            // \Laravel\Sanctum\Http\Middleware\EnsureFrontendRequestsAreStateful::class,
            ThrottleRequests::class . ':api',
            SubstituteBindings::class,
        ],
    ];

    /**
     * The application's middleware aliases.
     *
     * Aliases may be used to conveniently assign middleware to routes and groups.
     *
     * @var array<string, class-string|string>
     */
    protected $middlewareAliases = [
        'auth'                  => Authenticate::class,
        'auth.basic'            => AuthenticateWithBasicAuth::class,
        'auth.session'          => AuthenticateSession::class,
        'cache.headers'         => SetCacheHeaders::class,
        'can'                   => Authorize::class,
        'guest'                 => RedirectIfAuthenticated::class,
        'password.confirm'      => RequirePassword::class,
        'signed'                => ValidateSignature::class,
        'throttle'              => ThrottleRequests::class,
        'verified'              => EnsureEmailIsVerified::class,
        'admin'                 => AdminPermissionMiddleware::class,
        'is_not_demo'           => DemoCheckMiddleware::class,
        'newExtensionInstalled' => NewExtensionInstalled::class,
    ];

    /**
     * The application's route middleware.
     *
     * @var array
     */
    protected $routeMiddleware = [
        /**** OTHER MIDDLEWARES LOCALIZATION ****/
        'localize'                  => LaravelLocalizationRoutes::class,
        'localizationRedirect'      => LaravelLocalizationRedirectFilter::class,
        'localeSessionRedirect'     => LocaleSessionRedirect::class,
        'localeCookieRedirect'      => LocaleCookieRedirect::class,
        'localeViewPath'            => LaravelLocalizationViewPath::class,
        'checkInstallation'         => CheckInstallation::class,
        'custom'                    => Custom::class,
        'updateUserActivity'        => UpdateUserActivity::class,
        'sentry.context'            => SentryContextMiddleware::class,
        'surveyMiddleware'          => SurveyMiddleware::class,
    ];
}
