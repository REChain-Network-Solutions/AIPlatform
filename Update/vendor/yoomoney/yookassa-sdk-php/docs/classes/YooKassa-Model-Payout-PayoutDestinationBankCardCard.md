# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payout\PayoutDestinationBankCardCard
### Namespace: [\YooKassa\Model\Payout](../namespaces/yookassa-model-payout.md)
---
**Summary:**

Класс, представляющий модель PayoutDestinationBankCardCard.

**Description:**

Данные банковской карты.
Необходим при оплате PCI-DSS данными.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [ISO_3166_CODE_LENGTH](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#constant_ISO_3166_CODE_LENGTH) |  |  |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$card_type](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#property_card_type) |  | Тип банковской карты |
| public | [$cardType](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#property_cardType) |  | Тип банковской карты |
| public | [$first6](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#property_first6) |  | Первые 6 цифр номера карты |
| public | [$issuer_country](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#property_issuer_country) |  | Код страны, в которой выпущена карта |
| public | [$issuer_name](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#property_issuer_name) |  | Тип банковской карты |
| public | [$issuerCountry](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#property_issuerCountry) |  | Код страны, в которой выпущена карта |
| public | [$issuerName](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#property_issuerName) |  | Тип банковской карты |
| public | [$last4](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#property_last4) |  | Последние 4 цифры номера карты |

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
| public | [getCardType()](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#method_getCardType) |  | Возвращает тип банковской карты. |
| public | [getFirst6()](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#method_getFirst6) |  | Возвращает первые 6 цифр номера карты. |
| public | [getIssuerCountry()](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#method_getIssuerCountry) |  | Возвращает код страны, в которой выпущена карта. Передается в формате ISO-3166 alpha-2. |
| public | [getIssuerName()](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#method_getIssuerName) |  | Возвращает наименование банка, выпустившего карту. |
| public | [getLast4()](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#method_getLast4) |  | Возвращает последние 4 цифры номера карты. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setCardType()](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#method_setCardType) |  | Устанавливает тип банковской карты. |
| public | [setFirst6()](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#method_setFirst6) |  | Устанавливает первые 6 цифр номера карты. |
| public | [setIssuerCountry()](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#method_setIssuerCountry) |  | Устанавливает код страны, в которой выпущена карта. Передается в формате ISO-3166 alpha-2. |
| public | [setIssuerName()](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#method_setIssuerName) |  | Устанавливает наименование банка, выпустившего карту. |
| public | [setLast4()](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md#method_setLast4) |  | Устанавливает последние 4 цифры номера карты. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Payout/PayoutDestinationBankCardCard.php](../../lib/Model/Payout/PayoutDestinationBankCardCard.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Payout\PayoutDestinationBankCardCard

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_ISO_3166_CODE_LENGTH" class="anchor"></a>
###### ISO_3166_CODE_LENGTH
```php
ISO_3166_CODE_LENGTH = 2 : int
```



---
## Properties
<a name="property_card_type"></a>
#### public $card_type : string
---
***Description***

Тип банковской карты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_cardType"></a>
#### public $cardType : string
---
***Description***

Тип банковской карты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_first6"></a>
#### public $first6 : string
---
***Description***

Первые 6 цифр номера карты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_issuer_country"></a>
#### public $issuer_country : string
---
***Description***

Код страны, в которой выпущена карта

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_issuer_name"></a>
#### public $issuer_name : string
---
***Description***

Тип банковской карты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_issuerCountry"></a>
#### public $issuerCountry : string
---
***Description***

Код страны, в которой выпущена карта

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_issuerName"></a>
#### public $issuerName : string
---
***Description***

Тип банковской карты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_last4"></a>
#### public $last4 : string
---
***Description***

Последние 4 цифры номера карты

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


<a name="method_getCardType" class="anchor"></a>
#### public getCardType() : string|null

```php
public getCardType() : string|null
```

**Summary**

Возвращает тип банковской карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutDestinationBankCardCard](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md)

**Returns:** string|null - Тип банковской карты


<a name="method_getFirst6" class="anchor"></a>
#### public getFirst6() : string|null

```php
public getFirst6() : string|null
```

**Summary**

Возвращает первые 6 цифр номера карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutDestinationBankCardCard](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md)

**Returns:** string|null - Первые 6 цифр номера карты


<a name="method_getIssuerCountry" class="anchor"></a>
#### public getIssuerCountry() : string|null

```php
public getIssuerCountry() : string|null
```

**Summary**

Возвращает код страны, в которой выпущена карта. Передается в формате ISO-3166 alpha-2.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutDestinationBankCardCard](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md)

**Returns:** string|null - Код страны, в которой выпущена карта


<a name="method_getIssuerName" class="anchor"></a>
#### public getIssuerName() : string|null

```php
public getIssuerName() : string|null
```

**Summary**

Возвращает наименование банка, выпустившего карту.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutDestinationBankCardCard](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md)

**Returns:** string|null - Наименование банка, выпустившего карту


<a name="method_getLast4" class="anchor"></a>
#### public getLast4() : string|null

```php
public getLast4() : string|null
```

**Summary**

Возвращает последние 4 цифры номера карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutDestinationBankCardCard](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md)

**Returns:** string|null - Последние 4 цифры номера карты


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


<a name="method_setCardType" class="anchor"></a>
#### public setCardType() : self

```php
public setCardType(string|null $card_type = null) : self
```

**Summary**

Устанавливает тип банковской карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutDestinationBankCardCard](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | card_type  | Тип банковской карты |

**Returns:** self - 


<a name="method_setFirst6" class="anchor"></a>
#### public setFirst6() : self

```php
public setFirst6(string|null $first6 = null) : self
```

**Summary**

Устанавливает первые 6 цифр номера карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutDestinationBankCardCard](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | first6  | Первые 6 цифр номера карты |

**Returns:** self - 


<a name="method_setIssuerCountry" class="anchor"></a>
#### public setIssuerCountry() : self

```php
public setIssuerCountry(string|null $issuer_country = null) : self
```

**Summary**

Устанавливает код страны, в которой выпущена карта. Передается в формате ISO-3166 alpha-2.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutDestinationBankCardCard](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | issuer_country  | Код страны, в которой выпущена карта |

**Returns:** self - 


<a name="method_setIssuerName" class="anchor"></a>
#### public setIssuerName() : self

```php
public setIssuerName(string|null $issuer_name = null) : self
```

**Summary**

Устанавливает наименование банка, выпустившего карту.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutDestinationBankCardCard](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | issuer_name  | Наименование банка, выпустившего карту |

**Returns:** self - 


<a name="method_setLast4" class="anchor"></a>
#### public setLast4() : self

```php
public setLast4(string|null $last4 = null) : self
```

**Summary**

Устанавливает последние 4 цифры номера карты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\PayoutDestinationBankCardCard](../classes/YooKassa-Model-Payout-PayoutDestinationBankCardCard.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | last4  | Последние 4 цифры номера карты |

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