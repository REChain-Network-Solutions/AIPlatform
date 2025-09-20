<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Model\Receipt;

use DateTime;
use Exception;
use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Class IndustryDetails.
 *
 * Данные отраслевого реквизита
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $federalId Идентификатор федерального органа исполнительной власти (тег в 54 ФЗ — 1262)
 * @property string $federal_id Идентификатор федерального органа исполнительной власти (тег в 54 ФЗ — 1262)
 * @property DateTime $documentDate Дата документа основания (тег в 54 ФЗ — 1263)
 * @property DateTime $document_date Дата документа основания (тег в 54 ФЗ — 1263)
 * @property string $documentNumber Номер нормативного акта федерального органа исполнительной власти (тег в 54 ФЗ — 1264)
 * @property string $document_number Номер нормативного акта федерального органа исполнительной власти (тег в 54 ФЗ — 1264)
 * @property string $value Значение отраслевого реквизита (тег в 54 ФЗ — 1265)
 */
class IndustryDetails extends AbstractObject
{
    /** @var int Максимальная длинна номера документа */
    public const DOCUMENT_NUMBER_MAX_LENGTH = 32;

    /** @var int Максимальная длинна значение отраслевого реквизита */
    public const VALUE_MAX_LENGTH = 256;

    /** @var string Формат даты документа */
    public const DOCUMENT_DATE_FORMAT = 'Y-m-d';

    /**
     * @var string|null Идентификатор федерального органа исполнительной власти (тег в 54 ФЗ — 1262)
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Regex("/(^00[1-9]{1}$)|(^0[1-6]{1}[0-9]{1}$)|(^07[0-3]{1}$)/")]
    private ?string $_federal_id = null;

    /**
     * @var DateTime|null Дата документа основания (тег в 54 ФЗ — 1263). Передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601)
     */
    #[Assert\NotBlank]
    #[Assert\Date]
    #[Assert\Type('DateTime')]
    private ?DateTime $_document_date = null;

    /**
     * @var string|null Номер нормативного акта федерального органа исполнительной власти, регламентирующего порядок заполнения реквизита «значение отраслевого реквизита» (тег в 54 ФЗ — 1264)
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::DOCUMENT_NUMBER_MAX_LENGTH)]
    private ?string $_document_number = null;

    /**
     * @var string|null Значение отраслевого реквизита (тег в 54 ФЗ — 1265)
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: self::VALUE_MAX_LENGTH)]
    private ?string $_value = null;

    /**
     * Возвращает идентификатор федерального органа исполнительной власти.
     *
     * @return string|null Идентификатор федерального органа исполнительной власти
     */
    public function getFederalId(): ?string
    {
        return $this->_federal_id;
    }

    /**
     * Устанавливает идентификатор федерального органа исполнительной власти.
     *
     * @param string|null $federal_id Идентификатор федерального органа исполнительной власти
     *
     * @return IndustryDetails
     */
    public function setFederalId(?string $federal_id = null): self
    {
        $this->_federal_id = $this->validatePropertyValue('_federal_id', $federal_id);
        return $this;
    }

    /**
     * Возвращает дату документа основания.
     *
     * @return DateTime|null Дата документа основания
     */
    public function getDocumentDate(): ?DateTime
    {
        return $this->_document_date;
    }

    /**
     * Устанавливает дату документа основания.
     *
     * @param DateTime|string|null $document_date Дата документа основания
     * @return IndustryDetails
     * @throws Exception
     */
    public function setDocumentDate(DateTime|string|null $document_date = null): self
    {
        $this->_document_date = $this->validatePropertyValue('_document_date', $document_date);
        return $this;
    }

    /**
     * Возвращает номер нормативного акта федерального органа исполнительной власти.
     *
     * @return string|null Номер нормативного акта федерального органа исполнительной власти
     */
    public function getDocumentNumber(): ?string
    {
        return $this->_document_number;
    }

    /**
     * Устанавливает номер нормативного акта федерального органа исполнительной власти.
     *
     * @param string|null $document_number Номер нормативного акта федерального органа исполнительной власти
     * @return self
     */
    public function setDocumentNumber(?string $document_number = null): self
    {
        $this->_document_number = $this->validatePropertyValue('_document_number', $document_number);
        return $this;
    }

    /**
     * Возвращает значение отраслевого реквизита.
     *
     * @return string|null Значение отраслевого реквизита
     */
    public function getValue(): ?string
    {
        return $this->_value;
    }

    /**
     * Устанавливает значение отраслевого реквизита.
     *
     * @param string|null $value Значение отраслевого реквизита
     * @return IndustryDetails
     */
    public function setValue(?string $value = null): self
    {
        $this->_value = $this->validatePropertyValue('_value', $value);
        return $this;
    }

    /**
     * {@inheritdoc}
     */
    public function jsonSerialize(): array
    {
        $result = parent::jsonSerialize();
        $result['document_date'] = $this->getDocumentDate()->format(self::DOCUMENT_DATE_FORMAT);

        return $result;
    }
}
