def write_data_arff(data):
    print '''<?xml version="1.0" encoding="utf-8"?>
 
 <!DOCTYPE dataset
 [
    <!ELEMENT dataset (header,body)>
    <!ATTLIST dataset name CDATA #REQUIRED>
    <!ATTLIST dataset version CDATA "3.5.4">
 
    <!ELEMENT header (notes?,attributes)>
    <!ELEMENT body (instances)>
    <!ELEMENT notes ANY>   <!--  comments, information, copyright, etc. -->
 
    <!ELEMENT attributes (attribute+)>
    <!ELEMENT attribute (labels?,metadata?,attributes?)>
    <!ATTLIST attribute name CDATA #REQUIRED>
    <!ATTLIST attribute type (numeric|date|nominal|string|relational) #REQUIRED>
    <!ATTLIST attribute format CDATA #IMPLIED>
    <!ATTLIST attribute class (yes|no) "no">
    <!ELEMENT labels (label*)>   <!-- only for type "nominal" -->
    <!ELEMENT label ANY>
    <!ELEMENT metadata (property*)>
    <!ELEMENT property ANY>
    <!ATTLIST property name CDATA #REQUIRED>
 
    <!ELEMENT instances (instance*)>
    <!ELEMENT instance (value*)>
    <!ATTLIST instance type (normal|sparse) "normal">
    <!ATTLIST instance weight CDATA #IMPLIED>
    <!ELEMENT value (#PCDATA|instances)*>
    <!ATTLIST value index CDATA #IMPLIED>   <!-- 1-based index (only used for instance format "sparse") -->
    <!ATTLIST value missing (yes|no) "no">
 ]
 >'''
    print ''' <dataset name="iris" version="3.5.3">
    <header>
       <attributes>
          <attribute name="PostId" type="numeric"/>
          <attribute name="PostCreationDate" type="date"/>
          <attribute name="OwnerUserId" type="numeric"/>
          <attribute name="OwnerCreationDate" type="date"/>
          <attribute name="ReputationAtPostCreation" type="numeric"/>
          <attribute name="OwnerUndeletedAnswerCountAtPostTime" type="numeric"/>
          <attribute name="Title" type="string"/>
          <attribute name="BodyMarkdown" type="string"/>
          <attribute name="Tag1" type="string"/>
          <attribute name="Tag2" type="string"/>
          <attribute name="Tag3" type="string"/>
          <attribute name="Tag4" type="string"/>
          <attribute name="Tag5" type="string"/>
          <attribute name="PostClosedDate" type="date"/>
          <attribute name="OpenStatus" type="nominal"/>
       </attributes>
    </header>
 
    <body>
       <instances>'''
    
    for i in len(data):
        print '<instance>'
        for v in data.ix[i].values():
            print '<value>%s</value>' % v
        print '</instance>'

    print '''       </instances>
    </body>
 </dataset>'''
