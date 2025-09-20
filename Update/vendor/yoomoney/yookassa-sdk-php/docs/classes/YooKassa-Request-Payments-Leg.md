# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\Leg
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Класс, представляющий модель Leg.

**Description:**

Маршрут авиа перелета

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [ISO8601](../classes/YooKassa-Request-Payments-Leg.md#constant_ISO8601) |  | Формат даты. |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$carrier_code](../classes/YooKassa-Request-Payments-Leg.md#property_carrier_code) |  | Код авиакомпании по справочнику |
| public | [$carrierCode](../classes/YooKassa-Request-Payments-Leg.md#property_carrierCode) |  | Код авиакомпании по справочнику |
| public | [$departure_airport](../classes/YooKassa-Request-Payments-Leg.md#property_departure_airport) |  | Трёхбуквенный IATA-код аэропорта вылета |
| public | [$departure_date](../classes/YooKassa-Request-Payments-Leg.md#property_departure_date) |  | Дата вылета в формате YYYY-MM-DD ISO 8601:2004 |
| public | [$departureAirport](../classes/YooKassa-Request-Payments-Leg.md#property_departureAirport) |  | Трёхбуквенный IATA-код аэропорта вылета |
| public | [$departureDate](../classes/YooKassa-Request-Payments-Leg.md#property_departureDate) |  | Дата вылета в формате YYYY-MM-DD ISO 8601:2004 |
| public | [$destination_airport](../classes/YooKassa-Request-Payments-Leg.md#property_destination_airport) |  | Трёхбуквенный IATA-код аэропорта прилёта |
| public | [$destinationAirport](../classes/YooKassa-Request-Payments-Leg.md#property_destinationAirport) |  | Трёхбуквенный IATA-код аэропорта прилёта |

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
| public | [getCarrierCode()](../classes/YooKassa-Request-Payments-Leg.md#method_getCarrierCode) |  | Возвращает carrier_code. |
| public | [getDepartureAirport()](../classes/YooKassa-Request-Payments-Leg.md#method_getDepartureAirport) |  | Возвращает departure_airport. |
| public | [getDepartureDate()](../classes/YooKassa-Request-Payments-Leg.md#method_getDepartureDate) |  | Возвращает departure_date. |
| public | [getDestinationAirport()](../classes/YooKassa-Request-Payments-Leg.md#method_getDestinationAirport) |  | Возвращает destination_airport. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Request-Payments-Leg.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setCarrierCode()](../classes/YooKassa-Request-Payments-Leg.md#method_setCarrierCode) |  | Устанавливает carrier_code. |
| public | [setDepartureAirport()](../classes/YooKassa-Request-Payments-Leg.md#method_setDepartureAirport) |  | Устанавливает departure_airport. |
| public | [setDepartureDate()](../classes/YooKassa-Request-Payments-Leg.md#method_setDepartureDate) |  | Устанавливает departure_date. |
| public | [setDestinationAirport()](../classes/YooKassa-Request-Payments-Leg.md#method_setDestinationAirport) |  | Устанавливает destination_airport. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payments/Leg.php](../../lib/Request/Payments/Leg.php)
* Package: YooKassa\Request
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Request\Payments\Leg
* Implements:
  * [\YooKassa\Request\Payments\LegInterface](../classes/YooKassa-Request-Payments-LegInterface.md)

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
<a name="constant_ISO8601" class="anchor"></a>
###### ISO8601
Формат даты.

```php
ISO8601 = 'Y-m-d'
```



---
## Properties
<a name="property_carrier_code"></a>
#### public $carrier_code : string
---
***Description***

Код авиакомпании по справочнику

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_carrierCode"></a>
#### public $carrierCode : string
---
***Description***

Код авиакомпании по справочнику

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_departure_airport"></a>
#### public $departure_airport : string
---
***Description***

Трёхбуквенный IATA-код аэропорта вылета

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_departure_date"></a>
#### public $departure_date : string
---
***Description***

Дата вылета в формате YYYY-MM-DD ISO 8601:2004

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_departureAirport"></a>
#### public $departureAirport : string
---
***Description***

Трёхбуквенный IATA-код аэропорта вылета

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_departureDate"></a>
#### public $departureDate : string
---
***Description***

Дата вылета в формате YYYY-MM-DD ISO 8601:2004

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_destination_airport"></a>
#### public $destination_airport : string
---
***Description***

Трёхбуквенный IATA-код аэропорта прилёта

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_destinationAirport"></a>
#### public $destinationAirport : string
---
***Description***

Трёхбуквенный IATA-код аэропорта прилёта

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


<a name="method_getCarrierCode" class="anchor"></a>
#### public getCarrierCode() : string|null

```php
public getCarrierCode() : string|null
```

**Summary**

Возвращает carrier_code.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Leg](../classes/YooKassa-Request-Payments-Leg.md)

**Returns:** string|null - 


<a name="method_getDepartureAirport" class="anchor"></a>
#### public getDepartureAirport() : string|null

```php
public getDepartureAirport() : string|null
```

**Summary**

Возвращает departure_airport.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Leg](../classes/YooKassa-Request-Payments-Leg.md)

**Returns:** string|null - 


<a name="method_getDepartureDate" class="anchor"></a>
#### public getDepartureDate() : \DateTime|null

```php
public getDepartureDate() : \DateTime|null
```

**Summary**

Возвращает departure_date.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Leg](../classes/YooKassa-Request-Payments-Leg.md)

**Returns:** \DateTime|null - 


<a name="method_getDestinationAirport" class="anchor"></a>
#### public getDestinationAirport() : string|null

```php
public getDestinationAirport() : string|null
```

**Summary**

Возвращает destination_airport.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Leg](../classes/YooKassa-Request-Payments-Leg.md)

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
* Inherited From: [\YooKassa\Request\Payments\Leg](../classes/YooKassa-Request-Payments-Leg.md)

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


<a name="method_setCarrierCode" class="anchor"></a>
#### public setCarrierCode() : self

```php
public setCarrierCode(string|null $value = null) : self
```

**Summary**

Устанавливает carrier_code.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Leg](../classes/YooKassa-Request-Payments-Leg.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Код авиакомпании по справочнику [IATA](https://www.iata.org/publications/Pages/code-search.aspx). |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \Exception |  |

**Returns:** self - 


<a name="method_setDepartureAirport" class="anchor"></a>
#### public setDepartureAirport() : self

```php
public setDepartureAirport(string|null $value = null) : self
```

**Summary**

Устанавливает departure_airport.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Leg](../classes/YooKassa-Request-Payments-Leg.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Код аэропорта вылета по справочнику [IATA](https://www.iata.org/publications/Pages/code-search.aspx), например LED. |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \Exception |  |

**Returns:** self - 


<a name="method_setDepartureDate" class="anchor"></a>
#### public setDepartureDate() : self

```php
public setDepartureDate(\DateTime|string|null $value = null) : self
```

**Summary**

Устанавливает departure_date.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Leg](../classes/YooKassa-Request-Payments-Leg.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | value  | Дата вылета в формате YYYY-MM-DD по стандарту [ISO 8601:2004](http://www.iso.org/iso/catalogue_detail?csnumber=40874). |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \Exception |  |

**Returns:** self - 


<a name="method_setDestinationAirport" class="anchor"></a>
#### public setDestinationAirport() : self

```php
public setDestinationAirport(string|null $value = null) : self
```

**Summary**

Устанавливает destination_airport.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\Leg](../classes/YooKassa-Request-Payments-Leg.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Код аэропорта прилета по справочнику [IATA](https://www.iata.org/publications/Pages/code-search.aspx), например AMS. |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \Exception |  |

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