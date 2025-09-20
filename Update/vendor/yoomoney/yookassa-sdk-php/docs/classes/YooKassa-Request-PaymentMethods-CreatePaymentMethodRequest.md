# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest
### Namespace: [\YooKassa\Request\PaymentMethods](../namespaces/yookassa-request-paymentmethods.md)
---
**Summary:**

Класс, представляющий модель CreatePaymentMethodRequest.

**Description:**

Данные для проверки и сохранения способа оплаты.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$card](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#property_card) |  | Данные банковской карты (необходимы, если вы собираете данные карты пользователей на своей стороне). |
| public | [$client_ip](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#property_client_ip) |  | IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения. |
| public | [$clientIp](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#property_clientIp) |  | IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения. |
| public | [$confirmation](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#property_confirmation) |  | Данные, необходимые для инициирования сценария подтверждения привязки. |
| public | [$holder](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#property_holder) |  | Данные магазина, для которого сохраняется способ оплаты. |
| public | [$type](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#property_type) |  | Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [builder()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_builder) |  | Возвращает билдер объектов запросов создания платежа. |
| public | [clearValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_clearValidationError) |  | Очищает статус валидации текущего запроса. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getCard()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_getCard) |  | Возвращает card. |
| public | [getClientIp()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_getClientIp) |  | Возвращает client_ip. |
| public | [getConfirmation()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_getConfirmation) |  | Возвращает confirmation. |
| public | [getHolder()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_getHolder) |  | Возвращает holder. |
| public | [getLastValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_getLastValidationError) |  | Возвращает последнюю ошибку валидации. |
| public | [getType()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_getType) |  | Возвращает type. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [hasType()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_hasType) |  | Проверяет, были ли установлен код способа оплаты. |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setCard()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_setCard) |  | Устанавливает card. |
| public | [setClientIp()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_setClientIp) |  | Устанавливает client_ip. |
| public | [setConfirmation()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_setConfirmation) |  | Устанавливает confirmation. |
| public | [setHolder()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_setHolder) |  | Устанавливает holder. |
| public | [setType()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_setType) |  | Устанавливает type. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| public | [validate()](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md#method_validate) |  | Валидирует текущий запрос, проверяет все ли нужные свойства установлены. |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [setValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_setValidationError) |  | Устанавливает ошибку валидации. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/PaymentMethods/CreatePaymentMethodRequest.php](../../lib/Request/PaymentMethods/CreatePaymentMethodRequest.php)
* Package: YooKassa\Model
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)
  * \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest
* Implements:
  * [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestInterface](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequestInterface.md)

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
<a name="property_card"></a>
#### public $card : \YooKassa\Request\PaymentMethods\PaymentMethodCard
---
***Description***

Данные банковской карты (необходимы, если вы собираете данные карты пользователей на своей стороне).

**Type:** <a href="../classes/YooKassa-Request-PaymentMethods-PaymentMethodCard.html"><abbr title="\YooKassa\Request\PaymentMethods\PaymentMethodCard">PaymentMethodCard</abbr></a>

**Details:**


<a name="property_client_ip"></a>
#### public $client_ip : string
---
***Description***

IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_clientIp"></a>
#### public $clientIp : string
---
***Description***

IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_confirmation"></a>
#### public $confirmation : \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation
---
***Description***

Данные, необходимые для инициирования сценария подтверждения привязки.

**Type:** <a href="../classes/YooKassa-Request-PaymentMethods-ConfirmationData-AbstractConfirmation.html"><abbr title="\YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation">AbstractConfirmation</abbr></a>

**Details:**


<a name="property_holder"></a>
#### public $holder : \YooKassa\Request\PaymentMethods\PaymentMethodHolder
---
***Description***

Данные магазина, для которого сохраняется способ оплаты.

**Type:** <a href="../classes/YooKassa-Request-PaymentMethods-PaymentMethodHolder.html"><abbr title="\YooKassa\Request\PaymentMethods\PaymentMethodHolder">PaymentMethodHolder</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**



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


<a name="method_builder" class="anchor"></a>
#### public builder() : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestBuilder

```php
Static public builder() : \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestBuilder
```

**Summary**

Возвращает билдер объектов запросов создания платежа.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

**Returns:** \YooKassa\Request\PaymentMethods\CreatePaymentMethodRequestBuilder - Инстанс билдера объектов запросов


<a name="method_clearValidationError" class="anchor"></a>
#### public clearValidationError() : void

```php
public clearValidationError() : void
```

**Summary**

Очищает статус валидации текущего запроса.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

**Returns:** void - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array|\Traversable $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \Traversable</code> | sourceArray  | Ассоциативный массив с настройками |

**Returns:** void - 


<a name="method_getCard" class="anchor"></a>
#### public getCard() : \YooKassa\Request\PaymentMethods\PaymentMethodCard|null

```php
public getCard() : \YooKassa\Request\PaymentMethods\PaymentMethodCard|null
```

**Summary**

Возвращает card.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

**Returns:** \YooKassa\Request\PaymentMethods\PaymentMethodCard|null - 


<a name="method_getClientIp" class="anchor"></a>
#### public getClientIp() : string|null

```php
public getClientIp() : string|null
```

**Summary**

Возвращает client_ip.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

**Returns:** string|null - 


<a name="method_getConfirmation" class="anchor"></a>
#### public getConfirmation() : \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation|null

```php
public getConfirmation() : \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation|null
```

**Summary**

Возвращает confirmation.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

**Returns:** \YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation|null - 


<a name="method_getHolder" class="anchor"></a>
#### public getHolder() : \YooKassa\Request\PaymentMethods\PaymentMethodHolder|null

```php
public getHolder() : \YooKassa\Request\PaymentMethods\PaymentMethodHolder|null
```

**Summary**

Возвращает holder.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

**Returns:** \YooKassa\Request\PaymentMethods\PaymentMethodHolder|null - 


<a name="method_getLastValidationError" class="anchor"></a>
#### public getLastValidationError() : string|null

```php
public getLastValidationError() : string|null
```

**Summary**

Возвращает последнюю ошибку валидации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

**Returns:** string|null - Последняя произошедшая ошибка валидации


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает type.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

**Returns:** string|null - 


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_hasType" class="anchor"></a>
#### public hasType() : bool

```php
public hasType() : bool
```

**Summary**

Проверяет, были ли установлен код способа оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

**Returns:** bool - True если код способа оплаты был установлен, false если нет


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


<a name="method_setCard" class="anchor"></a>
#### public setCard() : self

```php
public setCard(\YooKassa\Request\PaymentMethods\PaymentMethodCard|array|null $card) : self
```

**Summary**

Устанавливает card.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\PaymentMethods\PaymentMethodCard OR array OR null</code> | card  | Данные для проверки и сохранения банковской карты. |

**Returns:** self - 


<a name="method_setClientIp" class="anchor"></a>
#### public setClientIp() : self

```php
public setClientIp(string|null $client_ip = null) : self
```

**Summary**

Устанавливает client_ip.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | client_ip  | IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения. |

**Returns:** self - 


<a name="method_setConfirmation" class="anchor"></a>
#### public setConfirmation() : self

```php
public setConfirmation(\YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation|array|null $confirmation = null) : self
```

**Summary**

Устанавливает confirmation.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation OR array OR null</code> | confirmation  | Данные, необходимые для инициирования сценария подтверждения привязки. |

**Returns:** self - 


<a name="method_setHolder" class="anchor"></a>
#### public setHolder() : self

```php
public setHolder(\YooKassa\Request\PaymentMethods\PaymentMethodHolder|array|null $holder = null) : self
```

**Summary**

Устанавливает holder.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\PaymentMethods\PaymentMethodHolder OR array OR null</code> | holder  | Данные магазина, для которого сохраняется способ оплаты. |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $type = null) : self
```

**Summary**

Устанавливает type.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта. |

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


<a name="method_validate" class="anchor"></a>
#### public validate() : bool

```php
public validate() : bool
```

**Summary**

Валидирует текущий запрос, проверяет все ли нужные свойства установлены.

**Details:**
* Inherited From: [\YooKassa\Request\PaymentMethods\CreatePaymentMethodRequest](../classes/YooKassa-Request-PaymentMethods-CreatePaymentMethodRequest.md)

**Returns:** bool - True если запрос валиден, false если нет


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


<a name="method_setValidationError" class="anchor"></a>
#### protected setValidationError() : void

```php
protected setValidationError(string $value) : void
```

**Summary**

Устанавливает ошибку валидации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | Ошибка, произошедшая при валидации объекта |

**Returns:** void - 


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