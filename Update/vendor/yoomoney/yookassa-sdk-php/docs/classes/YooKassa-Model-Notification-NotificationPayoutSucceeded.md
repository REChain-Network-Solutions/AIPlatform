# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Notification\NotificationPayoutSucceeded
### Namespace: [\YooKassa\Model\Notification](../namespaces/yookassa-model-notification.md)
---
**Summary:**

Класс объекта, присылаемого API при изменении статуса выплаты на "succeeded".


---
### Examples
Пример скрипта обработки уведомления

```php
require_once '../vendor/autoload.php';

try {
    $source = file_get_contents('php://input');
    $data = json_decode($source, true);

    $factory = new \YooKassa\Model\Notification\NotificationFactory();
    $notificationObject = $factory->factory($data);
    $responseObject = $notificationObject->getObject();

    $client = new \YooKassa\Client();

    if (!$client->isNotificationIPTrusted($_SERVER['REMOTE_ADDR'])) {
        header('HTTP/1.1 400 Something went wrong');

        exit;
    }

    if (\YooKassa\Model\Notification\NotificationEventType::PAYMENT_SUCCEEDED === $notificationObject->getEvent()) {
        $someData = [
            'paymentId' => $responseObject->getId(),
            'paymentStatus' => $responseObject->getStatus(),
        ];
    // Специфичная логика
    // ...
    } elseif (\YooKassa\Model\Notification\NotificationEventType::PAYMENT_WAITING_FOR_CAPTURE === $notificationObject->getEvent()) {
        $someData = [
            'paymentId' => $responseObject->getId(),
            'paymentStatus' => $responseObject->getStatus(),
        ];
    // Специфичная логика
    // ...
    } elseif (\YooKassa\Model\Notification\NotificationEventType::PAYMENT_CANCELED === $notificationObject->getEvent()) {
        $someData = [
            'paymentId' => $responseObject->getId(),
            'paymentStatus' => $responseObject->getStatus(),
        ];
    // Специфичная логика
    // ...
    } elseif (\YooKassa\Model\Notification\NotificationEventType::REFUND_SUCCEEDED === $notificationObject->getEvent()) {
        $someData = [
            'refundId' => $responseObject->getId(),
            'refundStatus' => $responseObject->getStatus(),
            'paymentId' => $responseObject->getPaymentId(),
        ];
    // ...
    // Специфичная логика
    } else {
        header('HTTP/1.1 400 Something went wrong');

        exit;
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

        exit;
    }
} catch (Exception $e) {
    header('HTTP/1.1 400 Something went wrong');

    exit;
}

```

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$event](../classes/YooKassa-Model-Notification-AbstractNotification.md#property_event) |  | Тип события |
| public | [$object](../classes/YooKassa-Model-Notification-NotificationPayoutSucceeded.md#property_object) |  | Объект с информацией о выплате |
| public | [$type](../classes/YooKassa-Model-Notification-AbstractNotification.md#property_type) |  | Тип уведомления в виде строки |
| protected | [$_event](../classes/YooKassa-Model-Notification-AbstractNotification.md#property__event) |  |  |
| protected | [$_type](../classes/YooKassa-Model-Notification-AbstractNotification.md#property__type) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Model-Notification-NotificationPayoutSucceeded.md#method_fromArray) |  | Конструктор объекта нотификации. |
| public | [getEvent()](../classes/YooKassa-Model-Notification-AbstractNotification.md#method_getEvent) |  | Возвращает тип события. |
| public | [getObject()](../classes/YooKassa-Model-Notification-NotificationPayoutSucceeded.md#method_getObject) |  | Возвращает объект с информацией о выплате, уведомление о которой хранится в текущем объекте. |
| public | [getType()](../classes/YooKassa-Model-Notification-AbstractNotification.md#method_getType) |  | Возвращает тип уведомления. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setObject()](../classes/YooKassa-Model-Notification-NotificationPayoutSucceeded.md#method_setObject) |  | Устанавливает объект с информацией о выплате, уведомление о которой хранится в текущем объекте. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [setEvent()](../classes/YooKassa-Model-Notification-AbstractNotification.md#method_setEvent) |  | Устанавливает тип события. |
| protected | [setType()](../classes/YooKassa-Model-Notification-AbstractNotification.md#method_setType) |  | Устанавливает тип уведомления. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Notification/NotificationPayoutSucceeded.php](../../lib/Model/Notification/NotificationPayoutSucceeded.php)
* Package: YooKassa\Model
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Model\Notification\AbstractNotification](../classes/YooKassa-Model-Notification-AbstractNotification.md)
  * \YooKassa\Model\Notification\NotificationPayoutSucceeded

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Properties
<a name="property_event"></a>
#### public $event : string
---
***Description***

Тип события

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Notification\AbstractNotification](../classes/YooKassa-Model-Notification-AbstractNotification.md)


<a name="property_object"></a>
#### public $object : \YooKassa\Model\Payout\PayoutInterface
---
***Description***

Объект с информацией о выплате

**Type:** <a href="../classes/YooKassa-Model-Payout-PayoutInterface.html"><abbr title="\YooKassa\Model\Payout\PayoutInterface">PayoutInterface</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип уведомления в виде строки

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Notification\AbstractNotification](../classes/YooKassa-Model-Notification-AbstractNotification.md)


<a name="property__event"></a>
#### protected $_event : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Тип произошедшего события
**Details:**
* Inherited From: [\YooKassa\Model\Notification\AbstractNotification](../classes/YooKassa-Model-Notification-AbstractNotification.md)


<a name="property__type"></a>
#### protected $_type : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>
Тип уведомления
**Details:**
* Inherited From: [\YooKassa\Model\Notification\AbstractNotification](../classes/YooKassa-Model-Notification-AbstractNotification.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

AbstractObject constructor.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  |  |

**Returns:** mixed - 


<a name="method___get" class="anchor"></a>
#### public __get() : mixed

```php
public __get(string $propertyName) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method___isset" class="anchor"></a>
#### public __isset() : bool

```php
public __isset(string $propertyName) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method___set" class="anchor"></a>
#### public __set() : void

```php
public __set(string $propertyName, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method___unset" class="anchor"></a>
#### public __unset() : void

```php
public __unset(string $propertyName) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array $sourceArray) : void
```

**Summary**

Конструктор объекта нотификации.

**Description**

Инициализирует текущий объект из ассоциативного массива, который просто путём JSON десериализации получен из
тела пришедшего запроса. При конструировании проверяется валидность типа передаваемого уведомления, если
передать уведомление не того типа, будет сгенерировано исключение типа {@link}

**Details:**
* Inherited From: [\YooKassa\Model\Notification\NotificationPayoutSucceeded](../classes/YooKassa-Model-Notification-NotificationPayoutSucceeded.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | sourceArray  | Ассоциативный массив с информацией об уведомлении |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \Exception|\YooKassa\Common\Exceptions\InvalidPropertyValueException | Генерируется если значение типа нотификации или события не равны "notification" и "payout.succeeded" соответственно, что может говорить о том, что переданные в конструктор данные не являются уведомлением нужного типа. |

**Returns:** void - 


<a name="method_getEvent" class="anchor"></a>
#### public getEvent() : string|null

```php
public getEvent() : string|null
```

**Summary**

Возвращает тип события.

**Description**

Тип события - одна из констант, указанных в перечислении {@link}.

**Details:**
* Inherited From: [\YooKassa\Model\Notification\AbstractNotification](../classes/YooKassa-Model-Notification-AbstractNotification.md)

**Returns:** string|null - Тип события


<a name="method_getObject" class="anchor"></a>
#### public getObject() : \YooKassa\Model\Payout\PayoutInterface

```php
public getObject() : \YooKassa\Model\Payout\PayoutInterface
```

**Summary**

Возвращает объект с информацией о выплате, уведомление о которой хранится в текущем объекте.

**Description**

Так как нотификация может быть сгенерирована и поставлена в очередь на отправку гораздо раньше, чем она будет
получена на сайте, то опираться на статус пришедшей выплаты не стоит, лучше запросить текущую информацию о
выплате у API.

**Details:**
* Inherited From: [\YooKassa\Model\Notification\NotificationPayoutSucceeded](../classes/YooKassa-Model-Notification-NotificationPayoutSucceeded.md)

**Returns:** \YooKassa\Model\Payout\PayoutInterface - Объект с информацией о выплате


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип уведомления.

**Description**

Тип уведомления - одна из констант, указанных в перечислении {@link}.

**Details:**
* Inherited From: [\YooKassa\Model\Notification\AbstractNotification](../classes/YooKassa-Model-Notification-AbstractNotification.md)

**Returns:** string|null - Тип уведомления в виде строки


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_offsetExists" class="anchor"></a>
#### public offsetExists() : bool

```php
public offsetExists(string $offset) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method_offsetGet" class="anchor"></a>
#### public offsetGet() : mixed

```php
public offsetGet(string $offset) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method_offsetSet" class="anchor"></a>
#### public offsetSet() : void

```php
public offsetSet(string $offset, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method_offsetUnset" class="anchor"></a>
#### public offsetUnset() : void

```php
public offsetUnset(string $offset) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_setObject" class="anchor"></a>
#### public setObject() : self

```php
public setObject(\YooKassa\Model\Payout\PayoutInterface|array $object) : self
```

**Summary**

Устанавливает объект с информацией о выплате, уведомление о которой хранится в текущем объекте.

**Details:**
* Inherited From: [\YooKassa\Model\Notification\NotificationPayoutSucceeded](../classes/YooKassa-Model-Notification-NotificationPayoutSucceeded.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Payout\PayoutInterface OR array</code> | object  |  |

**Returns:** self - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации
Является алиасом метода AbstractObject::jsonSerialize().

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_getUnknownProperties" class="anchor"></a>
#### protected getUnknownProperties() : array

```php
protected getUnknownProperties() : array
```

**Summary**

Возвращает массив свойств которые не существуют, но были заданы у объекта.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив с не существующими у текущего объекта свойствами


<a name="method_setEvent" class="anchor"></a>
#### protected setEvent() : self

```php
protected setEvent(string|null $event) : self
```

**Summary**

Устанавливает тип события.

**Details:**
* Inherited From: [\YooKassa\Model\Notification\AbstractNotification](../classes/YooKassa-Model-Notification-AbstractNotification.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | event  | Тип события |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### protected setType() : self

```php
protected setType(string|null $type) : self
```

**Summary**

Устанавливает тип уведомления.

**Details:**
* Inherited From: [\YooKassa\Model\Notification\AbstractNotification](../classes/YooKassa-Model-Notification-AbstractNotification.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип уведомления |

**Returns:** self - 


<a name="method_validatePropertyValue" class="anchor"></a>
#### protected validatePropertyValue() : mixed

```php
protected validatePropertyValue(string $propertyName, mixed $propertyValue) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  |  |
| <code lang="php">mixed</code> | propertyValue  |  |

**Returns:** mixed - 



---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney