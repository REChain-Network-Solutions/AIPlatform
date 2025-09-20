# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\FraudData
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Класс, представляющий модель FraudData.

**Description:**

Информация для проверки операции на мошенничество

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$merchant_customer_bank_account](../classes/YooKassa-Request-Payments-FraudData.md#property_merchant_customer_bank_account) |  | Данные банковского счета, открытого в вашей системе |
| public | [$merchantCustomerBankAccount](../classes/YooKassa-Request-Payments-FraudData.md#property_merchantCustomerBankAccount) |  | Данные банковского счета, открытого в вашей системе |
| public | [$topped_up_phone](../classes/YooKassa-Request-Payments-FraudData.md#property_topped_up_phone) |  | Номер телефона для пополнения |
| public | [$toppedUpPhone](../classes/YooKassa-Request-Payments-FraudData.md#property_toppedUpPhone) |  | Номер телефона для пополнения |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getMerchantCustomerBankAccount()](../classes/YooKassa-Request-Payments-FraudData.md#method_getMerchantCustomerBankAccount) |  | Возвращает данные банковского счета, открытого в вашей системе. |
| public | [getToppedUpPhone()](../classes/YooKassa-Request-Payments-FraudData.md#method_getToppedUpPhone) |  | Возвращает номер телефона для пополнения. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setMerchantCustomerBankAccount()](../classes/YooKassa-Request-Payments-FraudData.md#method_setMerchantCustomerBankAccount) |  | Устанавливает данные банковского счета, открытого в вашей системе. |
| public | [setToppedUpPhone()](../classes/YooKassa-Request-Payments-FraudData.md#method_setToppedUpPhone) |  | Устанавливает Номер телефона для пополнения. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payments/FraudData.php](../../lib/Request/Payments/FraudData.php)
* Package: YooKassa\Request
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Request\Payments\FraudData

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| deprecated |  | Больше не поддерживается. Вместо него нужно использовать `receiver`. |

---
## Properties
<a name="property_merchant_customer_bank_account"></a>
#### public $merchant_customer_bank_account : \YooKassa\Request\Payments\MerchantCustomerBankAccount|null
---
***Description***

Данные банковского счета, открытого в вашей системе

**Type:** <a href="../\YooKassa\Request\Payments\MerchantCustomerBankAccount|null"><abbr title="\YooKassa\Request\Payments\MerchantCustomerBankAccount|null">MerchantCustomerBankAccount|null</abbr></a>

**Details:**


<a name="property_merchantCustomerBankAccount"></a>
#### public $merchantCustomerBankAccount : \YooKassa\Request\Payments\MerchantCustomerBankAccount|null
---
***Description***

Данные банковского счета, открытого в вашей системе

**Type:** <a href="../\YooKassa\Request\Payments\MerchantCustomerBankAccount|null"><abbr title="\YooKassa\Request\Payments\MerchantCustomerBankAccount|null">MerchantCustomerBankAccount|null</abbr></a>

**Details:**


<a name="property_topped_up_phone"></a>
#### public $topped_up_phone : string|null
---
***Description***

Номер телефона для пополнения

**Type:** <a href="../string|null"><abbr title="string|null">string|null</abbr></a>

**Details:**


<a name="property_toppedUpPhone"></a>
#### public $toppedUpPhone : string|null
---
***Description***

Номер телефона для пополнения

**Type:** <a href="../string|null"><abbr title="string|null">string|null</abbr></a>

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


<a name="method_getMerchantCustomerBankAccount" class="anchor"></a>
#### public getMerchantCustomerBankAccount() : \YooKassa\Request\Payments\MerchantCustomerBankAccount|null

```php
public getMerchantCustomerBankAccount() : \YooKassa\Request\Payments\MerchantCustomerBankAccount|null
```

**Summary**

Возвращает данные банковского счета, открытого в вашей системе.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\FraudData](../classes/YooKassa-Request-Payments-FraudData.md)

**Returns:** \YooKassa\Request\Payments\MerchantCustomerBankAccount|null - Данные банковского счета


<a name="method_getToppedUpPhone" class="anchor"></a>
#### public getToppedUpPhone() : string|null

```php
public getToppedUpPhone() : string|null
```

**Summary**

Возвращает номер телефона для пополнения.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\FraudData](../classes/YooKassa-Request-Payments-FraudData.md)

**Returns:** string|null - 


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


<a name="method_setMerchantCustomerBankAccount" class="anchor"></a>
#### public setMerchantCustomerBankAccount() : self

```php
public setMerchantCustomerBankAccount(\YooKassa\Request\Payments\MerchantCustomerBankAccount|array|null $merchant_customer_bank_account = null) : self
```

**Summary**

Устанавливает данные банковского счета, открытого в вашей системе.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\FraudData](../classes/YooKassa-Request-Payments-FraudData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payments\MerchantCustomerBankAccount OR array OR null</code> | merchant_customer_bank_account  | Данные банковского счета |

**Returns:** self - 


<a name="method_setToppedUpPhone" class="anchor"></a>
#### public setToppedUpPhone() : self

```php
public setToppedUpPhone(string|null $topped_up_phone = null) : self
```

**Summary**

Устанавливает Номер телефона для пополнения.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\FraudData](../classes/YooKassa-Request-Payments-FraudData.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | topped_up_phone  | Номер телефона для пополнения. Не более 15 символов. Пример: ~`79110000000` |

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