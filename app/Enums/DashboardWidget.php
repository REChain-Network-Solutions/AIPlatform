<?php

declare(strict_types=1);

namespace App\Enums;

use App\Enums\Contracts\WithStringBackedEnum;
use App\Enums\Traits\EnumTo;
use App\Enums\Traits\StringBackedEnumTrait;

enum DashboardWidget: string implements WithStringBackedEnum
{
    use EnumTo;
    use StringBackedEnumTrait;

    case PREMIUM_ADVANTAGES = 'premium-advantages';
    case WHAT_IS_NEW = 'what-is-new';
    case USAGE_OVERVIEW = 'usage-overview';
    case FINANCE = 'finance';
    case REVENUE_SOURCE = 'revenue-source';
    case API_COST_DISTRIBUTION = 'api-cost-distribution';
    case TOP_COUNTRIES = 'top-countries';
    case COST_MANAGEMENT = 'cost-management';
    case NEW_CUSTOMERS = 'new-customers';
    case RECENT_TRANSACTIONS = 'recent-transactions';
    case USERS_AND_PLATFORM = 'users-and-platform';
    case USER_TRAFFIC = 'user-traffic';
    case POPULAR_AI_TOOLS = 'popular-ai-tools';
    case GENERATED_CONTENT = 'generated-content';
    case USERS = 'users';
    case USER_CLIENT = 'user-client';
    case RECENT_ACTIVITY = 'recent-activity';
    case SYSTEM_STATUS = 'system-status';

    public function label(): string
    {
        return match ($this) {
            self::API_COST_DISTRIBUTION 	=> __('Api Cost Distribution'),
            self::COST_MANAGEMENT 			    => __('Cost Management'),
            self::FINANCE 					          => __('Finance'),
            self::GENERATED_CONTENT 		   => __('Generated Content'),
            self::NEW_CUSTOMERS 			      => __('New Customers'),
            self::POPULAR_AI_TOOLS 			   => __('Popular Ai Tools'),
            self::PREMIUM_ADVANTAGES 		  => __('Premium Advantages'),
            self::RECENT_ACTIVITY 			    => __('Recent Activity'),
            self::RECENT_TRANSACTIONS 		 => __('Recent Transactions'),
            self::REVENUE_SOURCE 			     => __('Revenue Source'),
            self::SYSTEM_STATUS 			      => __('System Status'),
            self::TOP_COUNTRIES 			      => __('Top Countries'),
            self::USAGE_OVERVIEW 			     => __('Usage Overview'),
            self::USER_CLIENT 				       => __('User Client'),
            self::USER_TRAFFIC 				      => __('User Traffic'),
            self::USERS_AND_PLATFORM 		  => __('Users And Platform'),
            self::USERS 					            => __('Users'),
            self::WHAT_IS_NEW 				       => __('What Is New'),
        };
    }
}
