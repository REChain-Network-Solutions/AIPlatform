## Настройки SDK API ЮKassa

[Справочник API ЮKassa](https://yookassa.ru/developers/api)

С помощью этого SDK вы можете работать с онлайн-платежами: отправлять запросы на оплату,
сохранять платежную информацию для повторных списаний, совершать возвраты и многое другое.

* [Установка дополнительных настроек для Curl](#Установка-дополнительных-настроек-для-Curl)
* [Аутентификация](#Аутентификация)
* [Статистические данные об используемом окружении](#Статистические-данные-об-используемом-окружении)
* [Получение информации о магазине](#Получение-информации-о-магазине)
* [Работа с Webhook](#Работа-с-Webhook)
* [Входящие уведомления](#Входящие-уведомления)

---
### Установка дополнительных настроек для Curl <a name="Установка-дополнительных-настроек-для-Curl"></a>

Чтобы установить дополнительные настройки Curl, можно воспользоваться методом `setAdvancedCurlOptions` класса `\YooKassa\Client\CurlClient`, создав свой класс Curl клиента c наследованием от `\YooKassa\Client\CurlClient`. Далее можно переопределить метод `setAdvancedCurlOptions` и задать в нем установку своих параметров методом `setCurlOption`.

Создаем класс:
```php
class CustomCurlClient extends \YooKassa\Client\CurlClient
{
    public function setAdvancedCurlOptions()
    {
        $this->setCurlOption(CURLOPT_SSL_VERIFYPEER, false);
    }
}
```

И применяем новый класс:
```php
$client = new \YooKassa\Client(new CustomCurlClient());
$client->setAuth('xxxxxx', 'test_XXXXXXX');
```

---
### Аутентификация <a name="Аутентификация"></a>

Для работы с API необходимо прописать в конфигурации данные аутентификации. Существует два способа аутентификации:
- shopId + секретный ключ,
- OAuth-токен. [Подробнее в документации по API](https://yookassa.ru/developers/partners-api/basics)

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
// shopId + секретный ключ
$client->setAuth('xxxxxx', 'test_XXXXXXX');
// или OAuth-токен
$client->setAuthToken('token_XXXXXXX');
```

---

### Статистические данные об используемом окружении <a name="Статистические-данные-об-используемом-окружении"></a>

Для поддержки качества, а также для быстрого реагирования на ошибки SDK передает статистику в запросах к API ЮKassa.

По умолчанию SDK передает в запросах версию операционной системы, версию PHP, а также версию SDK. Но вы можете передать дополнительные данные об используемом фреймворке, CMS, а также о модуле в CMS.

Например, это может выглядеть так:
```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');
$userAgent = $client->getApiClient()->getUserAgent();
$userAgent->setFramework('Symfony', '5.2.1');
$userAgent->setCms('Symfony CMF', '1.2.1');
$userAgent->setModule('YooKassa', '1.0.0');
```

---

### Получение информации о магазине <a name="Получение-информации-о-магазине"></a>

После установки конфигурации можно проверить корректность данных, а также получить информацию о магазине.

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

try {
    $response = $client->me();
} catch (\Exception $e) {
    $response = $e;
}

var_dump($response);
```
В результате мы увидим примерно следующее:
```
array(5) {
  ["account_id"] => string(6) "XXXXXX"
  ["test"]=>  bool(true)
  ["fiscalization_enabled"]=> bool(true)
  ["payment_methods"]=> array(2) {
    [0]=> string(9) "yoo_money"
    [1]=> string(9) "bank_card"
  }
  ["status"]=> string(7) "enabled"
}
```
[Подробнее в документации по API](https://yookassa.ru/developers/api?lang=php#me_object)

---

### Работа с Webhook <a name="Работа-с-Webhook"></a>

Если вы подключаетесь к API через Oauth-токен, то можете настроить получение уведомлений о смене статуса платежа или возврата.

Например, ЮKassa может сообщить, когда объект платежа, созданный в вашем приложении, перейдет в статус `waiting_for_capture`.

В данном примере мы устанавливаем вебхуки для succeeded и canceled-уведомлений. А также проверяем, не установлены ли уже вебхуки. И если установлены на неверный адрес, удаляем.

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuthToken('token_XXXXXXX');

$webHookUrl = 'https://merchant-site.ru/payment-notification';
$needWebHookList = [
    \YooKassa\Model\Notification\NotificationEventType::PAYMENT_SUCCEEDED,
    \YooKassa\Model\Notification\NotificationEventType::PAYMENT_CANCELED,
];

try {
    $currentWebHookList = $client->getWebhooks()->getItems();
    foreach ($needWebHookList as $event) {
        $hookIsSet = false;
        foreach ($currentWebHookList as $webHook) {
            if ($webHook->getEvent() === $event) {
                if ($webHook->getUrl() !== $webHookUrl) {
                    $hookIsSet = false;
                    $client->removeWebhook($webHook->getId());
                } else {
                    $hookIsSet = true;
                }
            }
        }
        if (!$hookIsSet) {
            $client->addWebhook(['event' => $event, 'url' => $webHookUrl]);
        }
    }
} catch (Exception $e) {
    echo $e->getMessage();
}
var_dump($client->getWebhooks()->getItems());
```

В результате мы увидим примерно следующее:
```
array(2) {
  [0] => object(YooKassa\Model\Webhook\Webhook)#7 (4) {
    ["id":"YooKassa\Model\Webhook\Webhook":private] => string(39) "wh-52e51c6e-29a2-4a0d-854b-01cf022b5613"
    ["event":"YooKassa\Model\Webhook\Webhook":private] => string(16) "payment.canceled"
    ["url":"YooKassa\Model\Webhook\Webhook":private] => string(66) "https://merchant-site.ru/payment-notification"
  }
  [1] => object(YooKassa\Model\Webhook\Webhook)#8 (4) {
    ["id":"YooKassa\Model\Webhook\Webhook":private] => string(39) "wh-c331b3ee-fb65-428d-b128-1b837d9c4d93"
    ["event":"YooKassa\Model\Webhook\Webhook":private] => string(17) "payment.succeeded"
    ["url":"YooKassa\Model\Webhook\Webhook":private] => string(66) "https://merchant-site.ru/payment-notification"
  }
}
```
[Подробнее в документации по API](https://yookassa.ru/developers/api?lang=php#webhook)

### Входящие уведомления <a name="Входящие-уведомления"></a>

Если вы хотите отслеживать состояние платежей и возвратов, вы можете подписаться на уведомления ([webhook](#Работа-с-Webhook), callback).

Уведомления пригодятся в тех случаях, когда объект API изменяется без вашего участия. Например, если пользователю нужно подтвердить платеж, процесс оплаты может занять от нескольких минут до нескольких часов.
Вместо того, чтобы всё это время периодически отправлять GET-запросы, чтобы узнать статус платежа, вы можете просто дождаться уведомления от ЮKassa.

[Входящие уведомления в документации](https://yookassa.ru/developers/using-api/webhooks?lang=php)

#### Использование

Как только произойдет событие, на которое вы подписались, на URL, который вы указали при настройке, придет уведомление. В нем будут все данные об объекте на момент, когда произошло событие.

Вам нужно подтвердить, что вы получили уведомление. Для этого ответьте HTTP-кодом 200. ЮKassa проигнорирует всё,
что будет находиться в теле или в заголовках ответа. Ответы с любыми другими HTTP-кодами будут считаться невалидными, и ЮKassa продолжит доставлять уведомление в течение 24 часов, начиная с момента, когда событие произошло.

#### Пример обработки уведомления с помощью SDK

```php
require_once 'vendor/autoload.php';

try {
    $source = file_get_contents('php://input');
    $data = json_decode($source, true);

    $factory = new \YooKassa\Model\Notification\NotificationFactory();
    $notificationObject = $factory->factory($data);
    $responseObject = $notificationObject->getObject();
    
    $client = new \YooKassa\Client();

    if (!$client->isNotificationIPTrusted($_SERVER['REMOTE_ADDR'])) {
        header('HTTP/1.1 400 Something went wrong');
        exit();
    }

    if ($notificationObject->getEvent() === \YooKassa\Model\Notification\NotificationEventType::PAYMENT_SUCCEEDED) {
        $someData = [
            'paymentId' => $responseObject->getId(),
            'paymentStatus' => $responseObject->getStatus(),
        ];
        // Специфичная логика
        // ...
    } elseif ($notificationObject->getEvent() === \YooKassa\Model\Notification\NotificationEventType::PAYMENT_WAITING_FOR_CAPTURE) {
        $someData = [
            'paymentId' => $responseObject->getId(),
            'paymentStatus' => $responseObject->getStatus(),
        ];
        // Специфичная логика
        // ...
    } elseif ($notificationObject->getEvent() === \YooKassa\Model\Notification\NotificationEventType::PAYMENT_CANCELED) {
        $someData = [
            'paymentId' => $responseObject->getId(),
            'paymentStatus' => $responseObject->getStatus(),
        ];
        // Специфичная логика
        // ...
    } elseif ($notificationObject->getEvent() === \YooKassa\Model\Notification\NotificationEventType::REFUND_SUCCEEDED) {
        $someData = [
            'refundId' => $responseObject->getId(),
            'refundStatus' => $responseObject->getStatus(),
            'paymentId' => $responseObject->getPaymentId(),
        ];
        // ...
        // Специфичная логика
    } else {
        header('HTTP/1.1 400 Something went wrong');
        exit();
    }

    // Специфичная логика
    // ...

    $client->setAuth('xxxxxx', 'test_XXXXXXX');

    // Получим актуальную информацию о платеже
    if ($paymentInfo = $client->getPaymentInfo($someData['paymentId'])) {
        $paymentStatus = $paymentInfo->getStatus();
        // Специфичная логика
        // ...
    } else {
        header('HTTP/1.1 400 Something went wrong');
        exit();
    }

} catch (Exception $e) {
    header('HTTP/1.1 400 Something went wrong');
    exit();
}
```
