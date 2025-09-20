# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Invoices\CreateInvoiceRequest
### Namespace: [\YooKassa\Request\Invoices](../namespaces/yookassa-request-invoices.md)
---
**Summary:**

Класс, представляющий модель CreateInvoiceRequest.


---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$cart](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#property_cart) |  | Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой. |
| public | [$delivery_method_data](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#property_delivery_method_data) |  | Данные о способе доставки счета пользователю. Доступен только один способ — самостоятельная доставка: ЮKassa возвращает вам ссылку на счет, и вы передаете ее пользователю любым удобным для вас способом. |
| public | [$description](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#property_description) |  | Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37». |
| public | [$expires_at](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#property_expires_at) |  | Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2024-10-18T10:51:18.139Z` |
| public | [$locale](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#property_locale) |  | Язык интерфейса, писем и смс, которые будет видеть или получать пользователь. Формат соответствует [ISO/IEC 15897](https://en.wikipedia.org/wiki/Locale_(computer_software)). Возможные значения: ru_RU, en_US. Регистр важен. |
| public | [$metadata](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#property_metadata) |  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8. |
| public | [$payment_data](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#property_payment_data) |  | Данные для проведения платежа по выставленному счету. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [builder()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_builder) |  | Возвращает билдер объектов запросов создания платежа. |
| public | [clearValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_clearValidationError) |  | Очищает статус валидации текущего запроса. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getCart()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_getCart) |  | Возвращает cart. |
| public | [getDeliveryMethodData()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_getDeliveryMethodData) |  | Возвращает delivery_method_data. |
| public | [getDescription()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_getDescription) |  | Возвращает description. |
| public | [getExpiresAt()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_getExpiresAt) |  | Возвращает expires_at. |
| public | [getLastValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_getLastValidationError) |  | Возвращает последнюю ошибку валидации. |
| public | [getLocale()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_getLocale) |  | Возвращает locale. |
| public | [getMetadata()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_getMetadata) |  | Возвращает metadata. |
| public | [getPaymentData()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_getPaymentData) |  | Возвращает payment_data. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [hasCart()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_hasCart) |  | Проверяет, были ли установлены товары в корзине. |
| public | [hasDeliveryMethodData()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_hasDeliveryMethodData) |  | Проверяет, были ли установлены данные для о способе доставки счета. |
| public | [hasDescription()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_hasDescription) |  | Проверяет наличие описания транзакции в создаваемом счете. |
| public | [hasExpiresAt()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_hasExpiresAt) |  | Проверяет, был ли установлен срок действия счета. |
| public | [hasLocale()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_hasLocale) |  | Проверяет наличие языка интерфейса в создаваемом счете. |
| public | [hasMetadata()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_hasMetadata) |  | Проверяет, были ли установлены метаданные счета. |
| public | [hasPaymentData()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_hasPaymentData) |  | Проверяет, были ли установлены данные для проведения платежа. |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setCart()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_setCart) |  | Устанавливает cart. |
| public | [setDeliveryMethodData()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_setDeliveryMethodData) |  | Устанавливает delivery_method_data. |
| public | [setDescription()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_setDescription) |  | Устанавливает description. |
| public | [setExpiresAt()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_setExpiresAt) |  | Устанавливает expires_at. |
| public | [setLocale()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_setLocale) |  | Устанавливает locale. |
| public | [setMetadata()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_setMetadata) |  | Устанавливает metadata. |
| public | [setPaymentData()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_setPaymentData) |  | Устанавливает payment_data. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| public | [validate()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md#method_validate) |  | Проверяет на валидность текущий объект |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [setValidationError()](../classes/YooKassa-Common-AbstractRequest.md#method_setValidationError) |  | Устанавливает ошибку валидации. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Invoices/CreateInvoiceRequest.php](../../lib/Request/Invoices/CreateInvoiceRequest.php)
* Package: YooKassa\Model
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Common\AbstractRequest](../classes/YooKassa-Common-AbstractRequest.md)
  * \YooKassa\Request\Invoices\CreateInvoiceRequest
* Implements:
  * [\YooKassa\Request\Invoices\CreateInvoiceRequestInterface](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestInterface.md)

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
<a name="property_cart"></a>
#### public $cart : \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface
---
***Description***

Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.

**Type:** <a href="../\YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface"><abbr title="\YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface">ListObjectInterface</abbr></a>

**Details:**


<a name="property_delivery_method_data"></a>
#### public $delivery_method_data : \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData
---
***Description***

Данные о способе доставки счета пользователю. Доступен только один способ — самостоятельная доставка: ЮKassa возвращает вам ссылку на счет, и вы передаете ее пользователю любым удобным для вас способом.

**Type:** <a href="../classes/YooKassa-Request-Invoices-DeliveryMethodData-AbstractDeliveryMethodData.html"><abbr title="\YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData">AbstractDeliveryMethodData</abbr></a>

**Details:**


<a name="property_description"></a>
#### public $description : string
---
***Description***

Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37».

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_expires_at"></a>
#### public $expires_at : \DateTime
---
***Description***

Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2024-10-18T10:51:18.139Z`

**Type:** \DateTime

**Details:**


<a name="property_locale"></a>
#### public $locale : string
---
***Description***

Язык интерфейса, писем и смс, которые будет видеть или получать пользователь. Формат соответствует [ISO/IEC 15897](https://en.wikipedia.org/wiki/Locale_(computer_software)). Возможные значения: ru_RU, en_US. Регистр важен.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_metadata"></a>
#### public $metadata : \YooKassa\Model\Metadata
---
***Description***

Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.

**Type:** <a href="../classes/YooKassa-Model-Metadata.html"><abbr title="\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**


<a name="property_payment_data"></a>
#### public $payment_data : \YooKassa\Request\Invoices\PaymentData
---
***Description***

Данные для проведения платежа по выставленному счету.

**Type:** <a href="../classes/YooKassa-Request-Invoices-PaymentData.html"><abbr title="\YooKassa\Request\Invoices\PaymentData">PaymentData</abbr></a>

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
#### public builder() : \YooKassa\Request\Invoices\CreateInvoiceRequestBuilder

```php
Static public builder() : \YooKassa\Request\Invoices\CreateInvoiceRequestBuilder
```

**Summary**

Возвращает билдер объектов запросов создания платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** \YooKassa\Request\Invoices\CreateInvoiceRequestBuilder - Инстанс билдера объектов запросов


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


<a name="method_getCart" class="anchor"></a>
#### public getCart() : \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface

```php
public getCart() : \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает cart.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface - Корзина заказа — список товаров или услуг


<a name="method_getDeliveryMethodData" class="anchor"></a>
#### public getDeliveryMethodData() : \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData|null

```php
public getDeliveryMethodData() : \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData|null
```

**Summary**

Возвращает delivery_method_data.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** \YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData|null - Данные о способе доставки счета пользователю


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает description.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** string|null - Описание выставленного счета


<a name="method_getExpiresAt" class="anchor"></a>
#### public getExpiresAt() : \DateTime|null

```php
public getExpiresAt() : \DateTime|null
```

**Summary**

Возвращает expires_at.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** \DateTime|null - Срок действия счета


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


<a name="method_getLocale" class="anchor"></a>
#### public getLocale() : string|null

```php
public getLocale() : string|null
```

**Summary**

Возвращает locale.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** string|null - Язык интерфейса, писем и смс, которые будет видеть или получать пользователь


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает metadata.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** \YooKassa\Model\Metadata|null - Любые дополнительные данные


<a name="method_getPaymentData" class="anchor"></a>
#### public getPaymentData() : \YooKassa\Request\Invoices\PaymentData|null

```php
public getPaymentData() : \YooKassa\Request\Invoices\PaymentData|null
```

**Summary**

Возвращает payment_data.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** \YooKassa\Request\Invoices\PaymentData|null - Данные для проведения платежа по выставленному счету


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_hasCart" class="anchor"></a>
#### public hasCart() : bool

```php
public hasCart() : bool
```

**Summary**

Проверяет, были ли установлены товары в корзине.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** bool - True если товары в корзине были установлены, false если нет


<a name="method_hasDeliveryMethodData" class="anchor"></a>
#### public hasDeliveryMethodData() : bool

```php
public hasDeliveryMethodData() : bool
```

**Summary**

Проверяет, были ли установлены данные для о способе доставки счета.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** bool - True если данные о способе доставки счета были установлены, false если нет


<a name="method_hasDescription" class="anchor"></a>
#### public hasDescription() : bool

```php
public hasDescription() : bool
```

**Summary**

Проверяет наличие описания транзакции в создаваемом счете.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** bool - True если описание транзакции есть, false если нет


<a name="method_hasExpiresAt" class="anchor"></a>
#### public hasExpiresAt() : bool

```php
public hasExpiresAt() : bool
```

**Summary**

Проверяет, был ли установлен срок действия счета.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** bool - True если срок действия счета был установлен, false если нет


<a name="method_hasLocale" class="anchor"></a>
#### public hasLocale() : bool

```php
public hasLocale() : bool
```

**Summary**

Проверяет наличие языка интерфейса в создаваемом счете.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** bool - True если язык интерфейса есть, false если нет


<a name="method_hasMetadata" class="anchor"></a>
#### public hasMetadata() : bool

```php
public hasMetadata() : bool
```

**Summary**

Проверяет, были ли установлены метаданные счета.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** bool - True если метаданные были установлены, false если нет


<a name="method_hasPaymentData" class="anchor"></a>
#### public hasPaymentData() : bool

```php
public hasPaymentData() : bool
```

**Summary**

Проверяет, были ли установлены данные для проведения платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** bool - True если данные для проведения платежа были установлены, false если нет


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


<a name="method_setCart" class="anchor"></a>
#### public setCart() : self

```php
public setCart(\YooKassa\Common\ListObjectInterface|array|null $cart = null) : self
```

**Summary**

Устанавливает cart.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR array OR null</code> | cart  | Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой. |

**Returns:** self - 


<a name="method_setDeliveryMethodData" class="anchor"></a>
#### public setDeliveryMethodData() : self

```php
public setDeliveryMethodData(\YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData|array|null $delivery_method_data = null) : self
```

**Summary**

Устанавливает delivery_method_data.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData OR array OR null</code> | delivery_method_data  | Данные о способе доставки счета пользователю |

**Returns:** self - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description = null) : self
```

**Summary**

Устанавливает description.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37». |

**Returns:** self - 


<a name="method_setExpiresAt" class="anchor"></a>
#### public setExpiresAt() : self

```php
public setExpiresAt(\DateTime|string|null $expires_at = null) : self
```

**Summary**

Устанавливает expires_at.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | expires_at  | Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2024-10-18T10:51:18.139Z` |

**Returns:** self - 


<a name="method_setLocale" class="anchor"></a>
#### public setLocale() : self

```php
public setLocale(string|null $locale = null) : self
```

**Summary**

Устанавливает locale.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | locale  | Язык интерфейса, писем и смс, которые будет видеть или получать пользователь |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(\YooKassa\Model\Metadata|array|null $metadata = null) : self
```

**Summary**

Устанавливает metadata.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Metadata OR array OR null</code> | metadata  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8. |

**Returns:** self - 


<a name="method_setPaymentData" class="anchor"></a>
#### public setPaymentData() : self

```php
public setPaymentData(\YooKassa\Request\Invoices\PaymentData|array|null $payment_data = null) : self
```

**Summary**

Устанавливает payment_data.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Invoices\PaymentData OR array OR null</code> | payment_data  | Данные для проведения платежа по выставленному счету |

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

Проверяет на валидность текущий объект

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequest](../classes/YooKassa-Request-Invoices-CreateInvoiceRequest.md)

**Returns:** bool - True если объект запроса валиден, false если нет


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